#Question 7
#Input file is given
file=q7.txt
#Checking if the file exists
if [ -f $file ]; then
	echo "File $file exists"
else
	echo "File doesn't exist."
fi
#To save the output removing the duplicates create a new file
output_file=output.txt
#Removing the duplicate lines and save to the output file
sort $file | uniq > $output_file
echo "Duplicate lines are removed"
cat $output_file