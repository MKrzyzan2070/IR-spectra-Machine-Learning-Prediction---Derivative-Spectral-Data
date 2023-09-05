
#A list of all of the chemical species in this release of the NIST Chemistry WebBook can be downloaded from this page. 
#The list is in tab-delimited format can contains the following information:
#https://webbook.nist.gov/chemistry/download/

from progress.bar import Bar

def get_cas_nums_from_species_txt():
    import re
    doc = open("species.txt")
    species = doc.readlines()
    doc.close()
    cas_num_list = []
    for line in species:
        the_find_1 = re.findall("\t(.+)", line)
        if len(the_find_1) == 0:
            the_find_2 = re.findall("\t(.+)", the_find_1[0])
        elif len(the_find_1) > 0:
            the_find_2 = re.findall("\t(.+)", the_find_1[-1])
        else:
            continue
        cas_num_list.append(the_find_2[0])

    return cas_num_list

#https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=%227688-21-3+%22&Type=UV-vis

#https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=7688-21-3&Type=IR

def downloading(cas_num_list, number):
    cas_num = cas_num_list[number]
    check = False
    if cas_num != "N/A":
        import urllib.request
        url = "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=" + cas_num +"&Type=IR"
        urllib.request.urlretrieve(url, "IR_from_URL")
        check = True
    return check

def writing_to_the_file(cas_num_list):
   import os
   print("Starting the download...")
   
   progress_bar = Bar('Downloading the IR files: ', max=len(cas_num_list))
   
   for i in range(len(cas_num_list)):

       if downloading(cas_num_list, i) is True:
           try:
                doc = open("IR_from_URL")
                IR_data = doc.readlines()
                doc.close()
                os.remove("IR_from_URL")
                destination = open("All_the_IR", "a")
                destination.write("NEW NEW NEW\n")
                destination.writelines(IR_data)
                destination.write("END END END\n")
                destination.close()
           except:
                pass
       progress_bar.next()
   progress_bar.finish()


def clearing_All_the_IR():
    doc = open("All_the_IR", "w")
    doc.write("")
    print("done cleaning")


###THIS IS IMPORTANT VARIABLE:
cas_num_list = get_cas_nums_from_species_txt()
#####

writing_to_the_file(cas_num_list)

