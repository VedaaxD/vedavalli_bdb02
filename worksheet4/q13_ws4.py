import re
accessions=['xkn59438', 'yhdck2', 'eihd39d9', 'chdsye847', 'hedle3455', 'xjhd53e', '45da', 'de37dp']
def criterion1(accession):
    if '5' in accession:
        print(f"Contains the number 5 :{accession}")
def criterion2(accession):
    if re.search(r'[de]',accession):
            print(f"Contains the letter d/e {accession}")
def criterion3(accession):
    if re.search(r'd.*e',accession):
        print(f"Contains the letters d and e in the order:{accession}")
def criterion4(accession):
    if re.search(r'd.e',accession):
        print(f"Contains d and e with one letter in between them {accession}")
def criterion5(accession):
    if re.search(r'[de].*[de]',accession):
        print(f"Contains both d and e in any order {accession}")
def criterion6(accession):
    if accession.startswith(('x','y')):
        print(f"Starts with x or y: {accession}")
def criterion7(accession):
    if accession.startswith(('x','y')) and accession.endswith('e'):
        print(f"Starts with x or y and ends with e: {accession}")
def criterion8(accession):
    if re.search(r'\d{3,}',accession):
        print(f"Contains 3 or more digits in a row: {accession}")
def criterion9(accession):
    if re.search(r'd[arp]$',accession):
        print(f"Ends with d followed by a,r or p: {accession}")
def main():
    for accession in accessions:
        criterion1(accession)
        criterion2(accession)
        criterion3(accession)
        criterion4(accession)
        criterion5(accession)
        criterion6(accession)
        criterion7(accession)
        criterion8(accession)
        criterion9(accession)

if __name__=="__main__":
    main()

