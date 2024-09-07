#Question 8
#Initializing the threshold value
threshold=95
input_file=numbersq8.txt
output_file=threshold_numbers.txt
#Clearing the output file if it already exists
> $output_file
#Looping through every number from 1 to 100
while read -r number; do
	#Check if the number is greater than the threshold
	if (( number > threshold )); then
		echo "$number is greater than $threshold" >> $output_file
	fi
done < $input_file
#Display the contents of the file
cat $output_file