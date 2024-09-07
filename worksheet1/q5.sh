#Getting the total disk usage of the root directory
home_dir=$( du -s $HOME | awk '{print $1}')
#Geeting the toal disk usage of the home directory
root_dir=$( du -s / | awk '{print $1}' )
#Calculating the percentage of home usage relative to the root usage
percentage=$(echo "scale=2; ($home_dir / $root_dir) * 100" | bc)
#Displaying the result
echo "Home directory usage is $percentage% of the root directory usage"