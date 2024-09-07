#Question6
#Hardcoding the directory
directory=$HOME/test-dir
#File to store the empty folders
output_file="empty_folders.txt"
#Initializing the output file
> "output_file" 
#Finding all the empty subdirectories within the home directory
find "$directory" -type d -empty > "$output_file"
#Check if any empty directories were found
if [ -s "$output_file" ]; then
	echo "Empty subfolders have been listed in $output_file"
	cat empty_folders.txt
else
	echo "No empty subfolders found"
#Remove the created output file if there are no empty folders
	rm "$output_file" 
fi
