#Setting the threshold capacity of the disk usage in percentage
threshold=70
#Get the disk usage from the home directory
disk_usage=$(df -h "$HOME")
# Extract the percentage from the disk usage string
percentage=$(echo "$disk_usage" | cut -d ' ' -f 5 | sed 's/%//')
#Checking if the percentage of disk usage is greater than or equal to threshold 
if [[ "$percentage" -ge "$threshold" ]]; then
	#Printing a warning message
	echo "Warning: Disk space for $HOME is low"
else 
	echo "Disk space usage for the $HOME directory is below the threshold of ${threshold}%. No issues detected."
fi