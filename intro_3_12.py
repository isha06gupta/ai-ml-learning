
# 1. Sum of two numbers
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print("Sum:", a + b)


# 2. Even or Odd
n = int(input("Enter a number: "))
if n % 2 == 0:
    print("Even")
else:
    print("Odd")


# 3. Largest of three numbers
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

print("Largest number is:")

if (a > b):
    if (a > c):
        print(a)
    else:
        print(c)
elif (b > a):
    if (b > c):
        print(b)
    else:
        print(c)
else:
    if (c > a):
        print(c)
    else:
        print(a)

# 4. Simple Calculator
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
op = input("Enter operator (+, -, *, /): ")

if op == "+":
    print("Result:", num1 + num2)
elif op == "-":
    print("Result:", num1 - num2)
elif op == "*":
    print("Result:", num1 * num2)
elif op == "/":
    if num2 != 0:
        print("Result:", num1 / num2)
    else:
        print("Division by zero not allowed")
else:
    print("Invalid operator")



# 5. Factorial using loop
num= int(input("Enter a number:"))
fact =1
while num>0:
    fact=fact *num
    num=num-1
print("Factorial is : ", fact )


# 6. Multiplication table
n = int(input("Enter a number: "))
for i in range(1, 11):
    print(n, "x", i, "=", n * i)



# 7. Count vowels in a string
text = input("Enter a string: ").lower()
vowels = "aeiou"
count = 0

for ch in text:
    if ch in vowels:
        count += 1

print("Number of vowels:", count)


# 8. Palindrome check
s = input("Enter a string: ")
if s == s[::-1]:
    print("Palindrome")
else:
    print("Not a palindrome")



# 9. Sum of first n natural numbers
n = int(input("Enter n: "))
total = 0

for i in range(1, n + 1):
    total += i

print("Sum:", total)


# 10. List operations
lst = [1, 2, 3, 4]

lst.append(5)
print("After append:", lst)

lst.remove(2)
print("After remove:", lst)

print("Length of list:", len(lst))

print("All elements:")
for item in lst:
    print(item)


# 11. Simple Interest
p = float(input("Enter principal: "))
r = float(input("Enter rate: "))
t = float(input("Enter time: "))

si = (p * r * t) / 100
print("Simple Interest:", si)


# 12. Number guessing game
secret = 7
guess = 0

while guess != secret:
    guess = int(input("Guess the number (1-10): "))
    if guess < secret:
        print("Too low")
    elif guess > secret:
        print("Too high")

print("Correct guess!")


# 13. Fibonacci series
n = int(input("Enter number of terms: "))
a, b = 0, 1

for i in range(n):
    print(a, end=" ")
    a, b = b, a + b
print()


# 14. Count positive, negative, zeros
pos = neg = zero = 0

print("Enter 10 numbers:")
for i in range(10):
    n = int(input())
    if n > 0:
        pos += 1
    elif n < 0:
        neg += 1
    else:
        zero += 1

print("Positive:", pos)
print("Negative:", neg)
print("Zeros:", zero)
