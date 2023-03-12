DRAMP=open("/Users/michael/Documents/Programming 3/Science Fair 2020/Raw Data (DRAMP).txt", "r")
contents =DRAMP.read()
ADP = open("/Users/michael/Documents/Programming 3/Science Fair 2020/APD database raw.txt", "r")

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

# DRAMP
DRAMPsequences = []

string = ''
record = False
for k in range(len(contents)-3):
    if not record:
        if contents[k] == 'u' and contents[k+1] == "'":
            if contents[k+2].isupper() and contents[k+3].isupper():
                if contents[k+7].isalpha():
                    record = True
                    string = ''
                    string = string + contents[k]
    else:
        if contents[k] == "'":
            if contents[k + 1] == ':' or contents[k+1] == ',':
                record = False
                DRAMPsequences.append(string)
        else:
            string = string + contents[k]


for z in range(len(DRAMPsequences)):
    DRAMPsequences[z] = DRAMPsequences[z][1:]

# ADP
contents = ADP.read()
ADPsequences = []

string = ''
record = False
for k in range(len(contents)-3):
    if not record:
        if contents[k] == "\t" and contents[k+2].isalpha():
            record = True
            string = ''
            string = string + contents[k]
            print(k)
    else:
        if contents[k]+contents[k+1] == "\n ":
            record = False
            ADPsequences.append(string)
        else:
            string = string + contents[k]

for z in range(len(ADPsequences)):
    ADPsequences[z] = ADPsequences[z][1:]




combined = Remove(ADPsequences + DRAMPsequences)

with open("/Users/michael/Documents/Programming 3/Science Fair 2020/Combined Data", "w") as filehandle:
    filehandle.writelines("%s\n" % seq for seq in combined)