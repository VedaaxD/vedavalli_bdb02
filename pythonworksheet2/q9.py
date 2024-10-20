# Write a program to iterate over a dictionary and print key and values
foods = {
    'Cappuccino': 'Beverage',
    'Paella': 'Main Dish',
    'Kimchi': 'Side Dish',
    'Biryani': 'Main Dish',
    'Calamari': 'Side Dish',
    'Sushi': 'Main Dish',
    'Kebab': 'Main Dish',
    'Miso Soup': 'Side Dish',
    'Tacos': 'Main Dish',
    'Tea': 'Beverage',
    'Espresso': 'Beverage',
    'Falafel': 'Side Dish',
    'Pho': 'Main Dish',
    'Dim Sum': 'Side Dish'
}
def keys_and_values():
    print("The keys are:")
    for key in foods.keys():
        print(key)
    print("The values are:")
    for value in foods.values():
        print(value)
    print("The key-value pairs are:")
    for item in foods.items():
        print(item)
def main():
    keys_and_values()
if __name__=="__main__":
    main()