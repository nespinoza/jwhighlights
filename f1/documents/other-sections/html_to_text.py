import numpy as np


input_file = 'cycle3_html.txt'
output_file = 'cycle3.csv'

f = open(input_file, 'r')
fout = open(output_file, 'w')
output_string = ''
s = 0
while True:

    line = f.readline()
    if line != '':

        if '<tr>' in line:

            # Get info for proposal:
            proposal = {}

            # First item is always PID:
            line = f.readline()
            v = line.split('>')
            proposal['pid'] = v[2].split('<')[0]

            # Then, proposal title:
            line = f.readline()
            proposal['title'] = line.split('>')[1].split('</td')[0]
            # Remove any ; from here, as these are the CSV separators we use in our script:
            proposal['title'] = proposal['title'].replace(';',':')
            
            # Then, PIs. There's two cases. First, if single PI, easy::
            print(line)
            line = f.readline()
            if '</td>' in line:

                proposal['pis'] = line.split('PI:')[1].split('</td>')[0][1:]

            else:

                # If multiple PIs, keep reading the next line:
                proposal['pis'] = line.split('PI:')[1].split('<br /')[0][1:]
                line = f.readline() 
                proposal['pis'] += ','+line.split(':')[1].split('</td>')[0]

            # Now propietary period:
            line = f.readline()
            proposal['propietary period'] = line.split('>')[1].split('</td')[0]

            # Hours of telescope time:
            line = f.readline()
            proposal['hours'] = line.split('>')[1].split('</td')[0]
            # If that pesky / that was added in Cycle 2 is in, remove it:
            proposal['hours'] = proposal['hours'].split('/')[0]

            try:

                s += float(proposal['hours'])

            except:

                print('no sum:',proposal['pid'])
            # Instrument(s). If a single instrument, easy:
            line = f.readline()
            if '</td>' in line:

                proposal['instruments'] = line.split('>')[1].split('</td')[0]

            else:

                # If multiple instruments, iterate until the end:
                proposal['instruments'] = line.split('>')[1].split('<br /')[0]
                line = f.readline()

                while True:

                    if '</td>' in line:

                        proposal['instruments'] += ','+line.split('\t')[3].split('</td')[0]
                        break                    

                    else:

                        proposal['instruments'] += ','+line.split('\t')[3].split('<br /')[0]
                        line = f.readline()

            # Now write all this info to the file:
            for k in list(proposal.keys()):

                if k != 'instruments':

                    fout.write(proposal[k]+';')

                else:

                    fout.write(proposal[k]+'\n')

    else:

        break

print('total hours:',s)
fout.close()
f.close()
