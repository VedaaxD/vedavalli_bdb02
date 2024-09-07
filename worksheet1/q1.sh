#Question 1
#Defining the log files where the results will be stored
log_file="executable_files_list.txt"
#Searching for executable files
find / -type f -perm -u+x -print >> executable_files_list.txt
