import random
import math

def greet(name):
    print(f"Hello, {name}!")

def calculate_circle_area(radius):
    return math.pi * radius**2

def generate_random_list(length):
    return [random.randint(1, 100) for _ in range(length)]

def find_average(lst):
    return sum(lst) / len(lst)

def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True

def reverse_string(s):
    return s[::-1]

if __name__ == "__main__":
    name = input("Enter your name: ")
    greet(name)

    radius = float(input("Enter the radius of a circle: "))
    area = calculate_circle_area(radius)
    print(f"The area of the circle is: {area}")

    length = int(input("Enter the length of the random list: "))
    random_list = generate_random_list(length)
    print(f"Generated random list: {random_list}")
    average = find_average(random_list)
    print(f"The average of the list is: {average}")

    num = int(input("Enter a number to check if it's prime: "))
    if is_prime(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")

    string_to_reverse = input("Enter a string to reverse: ")
    reversed_string = reverse_string(string_to_reverse)
    print(f"The reversed string is: {reversed_string}")
