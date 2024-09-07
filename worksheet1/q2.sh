#Question 2
#File containing the data of the users
CSV_FILE=users.csv
#For reading the csv file
tail -n+2 "$CSV_FILE" | while IFS=',' read userID username_userdepartment; do
#splitting the input into username and userdepartment using the colon delimitter
	IFS=':' read username userdepartment <<< $username_userdepartment
#Printing the user information
	echo "userID: $userID"
	echo "username: $username"
	echo "userdepartment: $userdepartment"
	echo
done < users.csv
