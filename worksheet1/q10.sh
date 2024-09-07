#Question10
#hardcoding a dna sequence
DNA_sequence=AGCTAGCTGCTA
#function to count the nucleotides
count_nucleotides() {
	#Checking for any invalid characters present
	if [[ $DNA_sequence =~ [^ACGT] ]];then
		echo "Error. Invalid characters present."
		return 1
	fi
#Count the occurences of each nucleotide
a_count=$(echo "$DNA_sequence" | grep -o "A" | wc -l)
c_count=$(echo "$DNA_sequence" | grep -o "C" | wc -l)
g_count=$(echo "$DNA_sequence" | grep -o "G" | wc -l)
t_count=$(echo "$DNA_sequence" | grep -o "T" | wc -l) 
#Printing the output
echo "A: $a_count"
echo "C: $c_count"
echo "G: $g_count"
echo "T: $t_count"
}
#Calling the function 
count_nucleotides "$DNA_sequence"
