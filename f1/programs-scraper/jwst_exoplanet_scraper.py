
#!/usr/bin/env python3
"""
JWST exoplanet program + observation scraper

Part A
------
Scrape exoplanet-program tables from STScI Cycle GO pages.

Part B
------
Use those PIDs to fetch public APTX bundles and parse the internal XML into an
observation-level table. PDF parsing is kept only as a fallback/debug aid.

Key deterministic rules implemented
-----------------------------------
- split by distinct Observation Number
- split by explicit target name / TargetID in the APT XML
- split by optical element (filter / grating / disperser) when explicit
- no domain-expert inference for ambiguous multi-planet direct-imaging systems
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


USER_AGENT = "jwst-exoplanet-scraper/0.2 (local research use)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

CYCLE_URL_TEMPLATE = "https://www.stsci.edu/jwst/science-execution/approved-programs/general-observers/cycle-{cycle}-go"
APT_URL_TEMPLATE = "https://www.stsci.edu/jwst-program-info/download/jwst/apt/{pid}/"
PDF_URL_TEMPLATE = "https://www.stsci.edu/jwst-program-info/download/jwst/pdf/{pid}/"
VISIT_URL_TEMPLATE = (
    "https://www.stsci.edu/jwst-program-info/visits/"
    "?program={pid}&download=&pi=1&referrer=https://www.stsci.edu"
)
EXO_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

APT_NS = {"apt": "http://www.stsci.edu/JWST/APT"}

EXOPLANET_SECTION_HEADINGS = {
    1: ["Exoplanets and Exoplanet Formation"],
    2: ["Exoplanets and Exoplanet Formation"],
    3: ["Exoplanets and Exoplanet Formation"],
    4: [
        "Exoplanet Atmospheres and Habitability",
        "Exoplanetary System Formation and Dynamics",
        "Exoplanet System Formation and Dynamics",
    ],
    5: [
        "Exoplanet Atmospheres and Habitability",
        "Exoplanetary System Formation and Dynamics",
        "Exoplanet System Formation and Dynamics",
    ],
}

PROGRAM_COLUMNS = [
    "Cycle",
    "PID",
    "Title",
    "PIs",
    "Prop Period",
    "Hours",
    "Instruments",
    "Science Mode",
    "Sub-science theme",
]

OBS_COLUMNS = [
    "Cycle",
    "PID",
    "Star Name",
    "J-magnitude",
    "Planet Name",
    "Stellar Radius (Solar Radii)",
    "Stellar Mass (Solar Mass)",
    "Stellar Teff (K)",
    "Distance (pc)",
    "Planet Mass (Earth masses)",
    "Planet Radius (Earth radii)",
    "Planet Teq (K)",
    "Planet Period (days)",
    "Planet semi-major axis (AU)",
    "Observation",
    "Instrument/Mode",
    "Filter",
    "Science Mode",
    "Sub-science theme",
    "Seconds on target",
    "Completion status",
    "Comment on completion",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def request_bytes(url: str, timeout: int = 60) -> bytes:
    resp = SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def request_text(url: str, timeout: int = 60) -> str:
    resp = SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return resp.text


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return (
        s.replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
    )


def maybe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip().replace(",", "")
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def write_csv(rows: List[dict], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def write_xlsx(rows: List[dict], path: Path, columns: Sequence[str]) -> None:
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[list(columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")


# ------------------------- part A: cycle page scraping -------------------------

def get_relevant_headings_for_cycle(cycle: int) -> List[str]:
    return EXOPLANET_SECTION_HEADINGS.get(cycle, EXOPLANET_SECTION_HEADINGS[5])


def table_from_html_node(table: Tag) -> List[dict]:
    rows: List[dict] = []
    header_row = table.find("tr")
    if not header_row:
        return rows
    headers = [clean_text(th.get_text(" ", strip=True)) for th in header_row.find_all(["th", "td"])]
    if not headers:
        return rows
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [clean_text(td.get_text(" ", strip=True)) for td in cells]
        while len(vals) < len(headers):
            vals.append("")
        rows.append(dict(zip(headers, vals)))
    return rows


def find_section_tables(soup: BeautifulSoup, heading_texts: Sequence[str]) -> List[Tuple[str, Tag]]:
    found: List[Tuple[str, Tag]] = []
    heading_set = {clean_text(h).lower() for h in heading_texts}
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        title = clean_text(h.get_text(" ", strip=True))
        if title.lower() not in heading_set:
            continue
        nxt = h.next_sibling
        while nxt is not None:
            if isinstance(nxt, Tag) and re.match(r"^h[1-6]$", nxt.name or ""):
                break
            if isinstance(nxt, Tag):
                if nxt.name == "table":
                    found.append((title, nxt))
                for table in nxt.find_all("table"):
                    found.append((title, table))
            nxt = nxt.next_sibling
    uniq: List[Tuple[str, Tag]] = []
    seen = set()
    for title, table in found:
        if id(table) not in seen:
            uniq.append((title, table))
            seen.add(id(table))
    return uniq


def normalize_program_row(cycle: int, subtheme: str, row: dict) -> Optional[dict]:
    candidates = {clean_text(k).lower(): v for k, v in row.items()}
    pid = candidates.get("id") or candidates.get("pid") or candidates.get("program id")
    title = candidates.get("program title") or candidates.get("title")
    pis = candidates.get("pi & co-pis") or candidates.get("pi/co-pis") or candidates.get("pi and co-pis") or candidates.get("pi")
    prop_period = candidates.get("exclusive access period (months)") or candidates.get("exclusive access period") or candidates.get("proprietary period")
    hours = candidates.get("hours")
    instruments = candidates.get("instrument/ mode") or candidates.get("instrument / mode") or candidates.get("instrument/mode")

    if not pid or not title:
        return None
    pid = re.sub(r"[^\d]", "", str(pid))
    if not pid:
        return None

    row_join = " ".join(clean_text(str(v)) for v in row.values()).lower()
    if re.search(r"\barchival research\b", row_join):
        return None

    return {
        "Cycle": cycle,
        "PID": pid,
        "Title": clean_text(str(title)),
        "PIs": clean_text(str(pis or "")),
        "Prop Period": clean_text(str(prop_period or "")),
        "Hours": clean_text(str(hours or "")),
        "Instruments": clean_text(str(instruments or "")),
        "Science Mode": "",
        "Sub-science theme": clean_text(subtheme),
    }


def classify_program_science_mode(instruments: str) -> str:
    i = (instruments or "").lower()
    has_photo = any(k in i for k in ["imaging", "coron", "gts", "ami"])
    has_spec = any(k in i for k in ["soss", "bots", "ifu", "mrs", "lrs", "fs", "mos"])
    if has_photo and has_spec:
        return "Photometric/Spectroscopic"
    if has_photo:
        return "Photometric"
    if has_spec:
        return "Spectroscopic"
    return ""


def scrape_cycle_programs(cycle: int) -> List[dict]:
    html = request_text(CYCLE_URL_TEMPLATE.format(cycle=cycle))
    soup = BeautifulSoup(html, "html.parser")
    tables = find_section_tables(soup, get_relevant_headings_for_cycle(cycle))
    if not tables:
        raise RuntimeError(f"No relevant exoplanet tables found for cycle {cycle}")
    rows: List[dict] = []
    for subtheme, table in tables:
        for raw in table_from_html_node(table):
            norm = normalize_program_row(cycle, subtheme, raw)
            if norm:
                if not norm["Science Mode"]:
                    norm["Science Mode"] = classify_program_science_mode(norm["Instruments"])
                rows.append(norm)
    dedup: Dict[str, dict] = {}
    for row in rows:
        pid = row["PID"]
        if pid not in dedup:
            dedup[pid] = row
        elif row["Sub-science theme"] not in dedup[pid]["Sub-science theme"]:
            dedup[pid]["Sub-science theme"] += "; " + row["Sub-science theme"]
    return sorted(dedup.values(), key=lambda r: int(r["PID"]))


# ------------------------- part B: aptx/xml observation parsing -------------------------

KNOWN_MODE_MAP = {
    ("NIRSPEC", "NirspecBrightObjectTimeSeries"): "NIRSpec/BOTS",
    ("NIRSPEC", "NirspecIFUSpectroscopy"): "NIRSpec/IFU",
    ("NIRSPEC", "NirspecFixedSlitSpectroscopy"): "NIRSpec/FS",
    ("NIRSPEC", "NirspecMultiObjectSpectroscopy"): "NIRSpec/MOS",
    ("NIRSPEC", "NirspecMosSpectroscopy"): "NIRSpec/MOS",
    ("NIRISS", "NirissSoss"): "NIRISS/SOSS",
    ("NIRISS", "NirissAmi"): "NIRISS/AMI",
    ("NIRCAM", "NircamGrismTimeSeries"): "NIRCam/GTS",
    ("NIRCAM", "NircamCoronagraphy"): "NIRCam/Coronagraphy",
    ("NIRCAM", "NircamImaging"): "NIRCam/Imaging",
    ("MIRI", "MiriImaging"): "MIRI/Imaging",
    ("MIRI", "MiriLowResolutionSpectroscopy"): "MIRI/LRS",
    ("MIRI", "MiriLRS"): "MIRI/LRS",
    ("MIRI", "MiriMediumResolutionSpectroscopy"): "MIRI/MRS",
    ("MIRI", "MiriMRS"): "MIRI/MRS",
    ("MIRI", "MiriCoronagraphy"): "MIRI/Coronagraphy",
}

OPTICAL_KEYS = [
    "Filter", "FilterShort", "FilterLong", "Grating", "Disperser", "ReadoutPattern",
    "Wavelength", "Subarray", "CoronMask", "Aperture", "EtcFilter", "EtcGrating"
]


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def load_aptx_root_from_bytes(content: bytes) -> ET.Element:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml_members = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            if not xml_members:
                raise RuntimeError("No XML member found inside APTX zip")
            # prefer root proposal xml, typically <pid>.xml
            xml_members.sort(key=lambda n: ("/" in n, len(n)))
            xml_bytes = zf.read(xml_members[0])
            return ET.fromstring(xml_bytes)
    # fallback if server returns raw xml
    return ET.fromstring(content)


def fetch_aptx_root(pid: str) -> Optional[ET.Element]:
    url = APT_URL_TEMPLATE.format(pid=pid)
    try:
        content = request_bytes(url)
        return load_aptx_root_from_bytes(content)
    except Exception as e:
        log(f"[PID {pid}] APTX/XML fetch failed: {e}")
        return None


def fetch_pdf_text(pid: str) -> Optional[str]:
    if PdfReader is None:
        return None
    url = PDF_URL_TEMPLATE.format(pid=pid)
    try:
        content = request_bytes(url)
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages).strip() or None
    except Exception as e:
        log(f"[PID {pid}] PDF fallback failed: {e}")
        return None


def fetch_visit_status(pid: str) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    try:
        text = request_text(VISIT_URL_TEMPLATE.format(pid=pid))
    except Exception:
        return out
    try:
        root = ET.fromstring(text)
    except Exception:
        m = re.search(r"(<visitStatusReport[\s\S]*</visitStatusReport>)", text)
        if not m:
            return out
        root = ET.fromstring(m.group(1))
    for obs in root.findall(".//observation"):
        obsnum = clean_text(obs.findtext("observationNumber") or "")
        status = clean_text(obs.findtext("status") or "")
        comment = clean_text(obs.findtext("comment") or "")
        if obsnum:
            out[obsnum] = (status, comment)
    return out


def parse_target_map(root: ET.Element) -> Dict[str, str]:
    targets: Dict[str, str] = {}
    for t in root.findall(".//apt:Targets/apt:Target", APT_NS):
        number = clean_text(t.findtext("apt:Number", default="", namespaces=APT_NS))
        name = clean_text(t.findtext("apt:TargetName", default="", namespaces=APT_NS))
        if number and name:
            targets[number] = name
    return targets


def parse_template_fields(template_el: Optional[ET.Element]) -> Tuple[str, Dict[str, str]]:
    if template_el is None or len(template_el) == 0:
        return "", {}
    template = list(template_el)[0]
    template_name = strip_ns(template.tag)
    fields: Dict[str, str] = {}
    for el in template.iter():
        tag = strip_ns(el.tag)
        text = clean_text(el.text or "")
        if text and tag not in fields:
            fields[tag] = text
    return template_name, fields


def instrument_mode_from_template(instrument: str, template_name: str) -> str:
    instrument = (instrument or "").upper()
    if (instrument, template_name) in KNOWN_MODE_MAP:
        return KNOWN_MODE_MAP[(instrument, template_name)]
    if instrument == "NIRSPEC":
        if "TimeSeries" in template_name or "BrightObject" in template_name:
            return "NIRSpec/BOTS"
        if "IFU" in template_name:
            return "NIRSpec/IFU"
        if "FixedSlit" in template_name:
            return "NIRSpec/FS"
        if "MOS" in template_name or "Mos" in template_name:
            return "NIRSpec/MOS"
    if instrument == "NIRISS":
        if "Soss" in template_name or "SOSS" in template_name:
            return "NIRISS/SOSS"
        if "Ami" in template_name or "AMI" in template_name:
            return "NIRISS/AMI"
    if instrument == "NIRCAM":
        if "GrismTimeSeries" in template_name:
            return "NIRCam/GTS"
        if "Coron" in template_name:
            return "NIRCam/Coronagraphy"
        return "NIRCam/Imaging"
    if instrument == "MIRI":
        if "MediumResolution" in template_name or template_name == "MiriMRS":
            return "MIRI/MRS"
        if "LowResolution" in template_name or template_name == "MiriLRS":
            return "MIRI/LRS"
        if "Coron" in template_name:
            return "MIRI/Coronagraphy"
        return "MIRI/Imaging"
    return f"{instrument}/{template_name}" if instrument and template_name else ""


def extract_optical_elements(fields: Dict[str, str]) -> List[str]:
    vals: List[str] = []
    for key in OPTICAL_KEYS:
        val = clean_text(fields.get(key, ""))
        if val:
            vals.append(val)
    # If template stores both grating and filter, preserve both as separate rows
    exploded: List[str] = []
    for val in vals:
        if "/" in val and not val.startswith("NIR") and not val.startswith("MIRI"):
            exploded.extend([v.strip() for v in val.split("/") if v.strip()])
        else:
            exploded.append(val)
    uniq: List[str] = []
    seen = set()
    for v in exploded:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq or [""]


def parse_observation_xml(root: ET.Element) -> List[dict]:
    target_map = parse_target_map(root)
    rows: List[dict] = []

    for obs in root.findall(".//apt:Observation", APT_NS):
        obsnum = clean_text(obs.findtext("apt:Number", default="", namespaces=APT_NS))
        target_id = clean_text(obs.findtext("apt:TargetID", default="", namespaces=APT_NS))
        instrument = clean_text(obs.findtext("apt:Instrument", default="", namespaces=APT_NS)).upper()
        science_duration = clean_text(obs.findtext("apt:ScienceDuration", default="", namespaces=APT_NS))

        template_el = obs.find("apt:Template", APT_NS)
        template_name, fields = parse_template_fields(template_el)

        target_name = target_id
        m = re.match(r"^(\d+)\s+(.+)$", target_id)
        if m and m.group(1) in target_map:
            target_name = target_map[m.group(1)]
        elif target_id in target_map:
            target_name = target_map[target_id]

        rows.append({
            "Observation": obsnum,
            "Target Name": target_name,
            "Instrument": instrument,
            "ScienceDuration": maybe_float(science_duration),
            "TemplateName": template_name,
            "Instrument/Mode": instrument_mode_from_template(instrument, template_name),
            "OpticalElements": extract_optical_elements(fields),
            "TemplateFields": fields,
        })
    return rows


def split_target_name(target_name: str) -> Tuple[str, str]:
    target_name = clean_text(target_name)
    if not target_name:
        return "", ""
    m = re.match(r"^(.*\S)\s+([bcdefghij])$", target_name)
    if m:
        return clean_text(m.group(1)), target_name
    if re.match(r"^HR 8799 [bcde]$", target_name):
        return "HR 8799", target_name
    return target_name, ""


def query_exoplanet_archive(planet_name: str = "", host_name: str = "") -> dict:
    if not planet_name and not host_name:
        return {}
    if planet_name:
        safe_planet = planet_name.replace("'", "''")
        where = f"pl_name = '{safe_planet}'"
    else:
        safe_host = host_name.replace("'", "''")
        where = f"hostname = '{safe_host}'"
    query = f"""
    select top 1
        hostname, pl_name, sy_jmag, st_rad, st_mass, st_teff, sy_dist,
        pl_bmasse, pl_rade, pl_eqt, pl_orbper, pl_orbsmax
    from pscomppars
    where {where}
    """
    try:
        resp = SESSION.get(EXO_TAP_URL, params={"query": query, "format": "json"}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else {}
    except Exception:
        return {}


def infer_science_mode(title: str, subtheme: str, instrument_mode: str, template_name: str, pdf_text: str = "") -> str:
    t = " ".join([title or "", subtheme or "", instrument_mode or "", template_name or "", pdf_text or ""]).lower()
    if "phase curve" in t:
        return "Phase Curve"
    if "secondary eclipse" in t or re.search(r"\beclipse\b", t):
        return "Secondary Eclipse"
    if "transit" in t or "transmission" in t or "soss" in t or "bots" in t or "grism time series" in t:
        return "Transit"
    if "eclipse mapping" in t:
        return "Eclipse Mapping"
    if "coron" in t or "direct imag" in t or "ami" in t:
        return "Direct Imaging"
    if any(k in instrument_mode.lower() for k in ["soss", "bots", "ifu", "mrs", "lrs", "fs", "mos"]):
        return "Spectroscopic"
    if any(k in instrument_mode.lower() for k in ["imaging", "coron", "gts", "ami"]):
        return "Photometric"
    return ""


@dataclass
class ObsRow:
    Cycle: int
    PID: str
    Star_Name: str = ""
    J_magnitude: Optional[float] = None
    Planet_Name: str = ""
    Stellar_Radius_Solar_Radii: Optional[float] = None
    Stellar_Mass_Solar_Mass: Optional[float] = None
    Stellar_Teff_K: Optional[float] = None
    Distance_pc: Optional[float] = None
    Planet_Mass_Earth_masses: Optional[float] = None
    Planet_Radius_Earth_radii: Optional[float] = None
    Planet_Teq_K: Optional[float] = None
    Planet_Period_days: Optional[float] = None
    Planet_semimajor_axis_AU: Optional[float] = None
    Observation: str = ""
    Instrument_Mode: str = ""
    Filter: str = ""
    Science_Mode: str = ""
    Sub_science_theme: str = ""
    Seconds_on_target: Optional[float] = None
    Completion_status: str = ""
    Comment_on_completion: str = ""

    def to_output(self) -> dict:
        return {
            "Cycle": self.Cycle,
            "PID": self.PID,
            "Star Name": self.Star_Name,
            "J-magnitude": self.J_magnitude,
            "Planet Name": self.Planet_Name,
            "Stellar Radius (Solar Radii)": self.Stellar_Radius_Solar_Radii,
            "Stellar Mass (Solar Mass)": self.Stellar_Mass_Solar_Mass,
            "Stellar Teff (K)": self.Stellar_Teff_K,
            "Distance (pc)": self.Distance_pc,
            "Planet Mass (Earth masses)": self.Planet_Mass_Earth_masses,
            "Planet Radius (Earth radii)": self.Planet_Radius_Earth_radii,
            "Planet Teq (K)": self.Planet_Teq_K,
            "Planet Period (days)": self.Planet_Period_days,
            "Planet semi-major axis (AU)": self.Planet_semimajor_axis_AU,
            "Observation": self.Observation,
            "Instrument/Mode": self.Instrument_Mode,
            "Filter": self.Filter,
            "Science Mode": self.Science_Mode,
            "Sub-science theme": self.Sub_science_theme,
            "Seconds on target": self.Seconds_on_target,
            "Completion status": self.Completion_status,
            "Comment on completion": self.Comment_on_completion,
        }


def parse_observations_for_program(program_row: dict, include_completion: bool = False) -> List[dict]:
    pid = str(program_row["PID"])
    cycle = int(program_row["Cycle"])
    title = program_row.get("Title", "")
    subtheme = program_row.get("Sub-science theme", "")
    root = fetch_aptx_root(pid)
    if root is None:
        return []

    obs_items = parse_observation_xml(root)
    visit_map = fetch_visit_status(pid) if include_completion else {}
    pdf_text = None

    out_rows: List[dict] = []
    exo_cache: Dict[Tuple[str, str], dict] = {}

    for item in obs_items:
        star_name, planet_name = split_target_name(item["Target Name"])
        cache_key = (planet_name, star_name)
        if cache_key not in exo_cache:
            exo_cache[cache_key] = query_exoplanet_archive(planet_name=planet_name, host_name=star_name)
        exo = exo_cache[cache_key]

        completion_status, completion_comment = visit_map.get(item["Observation"], ("", ""))

        science_mode = infer_science_mode(
            title=title,
            subtheme=subtheme,
            instrument_mode=item["Instrument/Mode"],
            template_name=item["TemplateName"],
            pdf_text=pdf_text or "",
        )
        if not science_mode and not pdf_text:
            pdf_text = fetch_pdf_text(pid) or ""
            science_mode = infer_science_mode(
                title=title,
                subtheme=subtheme,
                instrument_mode=item["Instrument/Mode"],
                template_name=item["TemplateName"],
                pdf_text=pdf_text,
            )

        for optical in item["OpticalElements"]:
            row = ObsRow(
                Cycle=cycle,
                PID=pid,
                Star_Name=clean_text(exo.get("hostname") or star_name),
                J_magnitude=exo.get("sy_jmag"),
                Planet_Name=clean_text(exo.get("pl_name") or planet_name),
                Stellar_Radius_Solar_Radii=exo.get("st_rad"),
                Stellar_Mass_Solar_Mass=exo.get("st_mass"),
                Stellar_Teff_K=exo.get("st_teff"),
                Distance_pc=exo.get("sy_dist"),
                Planet_Mass_Earth_masses=exo.get("pl_bmasse"),
                Planet_Radius_Earth_radii=exo.get("pl_rade"),
                Planet_Teq_K=exo.get("pl_eqt"),
                Planet_Period_days=exo.get("pl_orbper"),
                Planet_semimajor_axis_AU=exo.get("pl_orbsmax"),
                Observation=item["Observation"],
                Instrument_Mode=item["Instrument/Mode"],
                Filter=optical,
                Science_Mode=science_mode,
                Sub_science_theme=subtheme,
                Seconds_on_target=item["ScienceDuration"],
                Completion_status=completion_status,
                Comment_on_completion=completion_comment,
            )
            out_rows.append(row.to_output())

    seen = set()
    dedup: List[dict] = []
    for row in out_rows:
        key = (
            row["PID"],
            row["Observation"],
            row["Star Name"],
            row["Planet Name"],
            row["Instrument/Mode"],
            row["Filter"],
            row["Seconds on target"],
        )
        if key not in seen:
            dedup.append(row)
            seen.add(key)
    return dedup


def scrape_observations_for_programs(program_rows: List[dict], include_completion: bool = False) -> List[dict]:
    rows: List[dict] = []
    for i, prow in enumerate(program_rows, start=1):
        log(f"[{i}/{len(program_rows)}] PID {prow['PID']} {prow.get('Title','')[:80]}")
        try:
            rows.extend(parse_observations_for_program(prow, include_completion=include_completion))
        except Exception as e:
            log(f"[PID {prow['PID']}] ERROR: {e}")
        time.sleep(0.2)
    return rows


def parse_cycles(cycles_arg: str) -> List[int]:
    cycles = []
    for part in cycles_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cycles.extend(range(int(a), int(b) + 1))
        else:
            cycles.append(int(part))
    return sorted(set(cycles))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape JWST exoplanet programs and observations.")
    ap.add_argument("--cycles", default="1-5", help="e.g. '1-5' or '4,5'")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--skip-programs", action="store_true")
    ap.add_argument("--skip-observations", action="store_true")
    ap.add_argument("--include-completion", action="store_true")
    ap.add_argument("--programs-csv", default="", help="Optional existing program CSV for part B")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_programs: List[dict] = []
    if args.programs_csv:
        all_programs = pd.read_csv(args.programs_csv).to_dict(orient="records")
    elif not args.skip_programs:
        for cycle in parse_cycles(args.cycles):
            log(f"Scraping cycle {cycle} program page...")
            all_programs.extend(scrape_cycle_programs(cycle))
        write_csv(all_programs, outdir / "jwst_exoplanet_programs.csv", PROGRAM_COLUMNS)
        write_xlsx(all_programs, outdir / "jwst_exoplanet_programs.xlsx", PROGRAM_COLUMNS)
        log(f"Wrote {len(all_programs)} program rows")

    if args.skip_observations:
        return

    if not all_programs:
        raise SystemExit("No programs available. Use part A first or pass --programs-csv")

    obs_rows = scrape_observations_for_programs(all_programs, include_completion=args.include_completion)
    write_csv(obs_rows, outdir / "jwst_exoplanet_observations.csv", OBS_COLUMNS)
    write_xlsx(obs_rows, outdir / "jwst_exoplanet_observations.xlsx", OBS_COLUMNS)
    log(f"Wrote {len(obs_rows)} observation rows")


if __name__ == "__main__":
    main()
