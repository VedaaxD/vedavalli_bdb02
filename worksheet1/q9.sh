#Function to greet based on time
greet_based_on_time() {
	#Getting the current time
	current_hour=$(date +%H)
	#Giving the greeting based on the current hour
	if [[ current_hour -ge 12 && current_hour -lt 18 ]];then
		echo "Good afternoon!"
	fi
	if [[ current_hour -ge 18 || current_hour -lt 5 ]];then
		echo "Good night!"
    else
	echo "Good morning!"
fi
}
#Callng the function
greet_based_on_time