#1.Write a Python program to find the sum of all even and odd elements separately in a list.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

even_sum = 0
odd_sum = 0

for num in numbers:
    if num % 2 == 0:
        even_sum += num
    else:
        odd_sum += num

print("Sum of even numbers:", even_sum)
print("Sum of odd numbers:", odd_sum)

#2. Write a program that takes a string as input and prints the frequency of each character
text = input("Enter a string: ")

freq = {}

for ch in text:
    if ch in freq:
        freq[ch] += 1
    else:
        freq[ch] = 1

for ch in freq:
    print(ch, ":", freq[ch])
#3. Write a program that takes a sentence or multiple sentences as input and prints the frequency of each word.
text = input("Enter text: ")

words = text.lower().split()
freq = {}

for word in words:
    freq[word] = freq.get(word, 0) + 1

for word, count in freq.items():
    print(word, ":", count)

#4. Write a program to find the first and the second largest element of a list without using built-in sorting functions.
nums = [10, 5, 20, 8, 15]

first = second = float('-inf')

for num in nums:
    if num > first:
        second = first
        first = num
    elif num > second and num != first:
        second = num

print("First largest:", first)
print("Second largest:", second)


#5. Write a program to remove duplicate elements from a list keeping the original order.
nums = [1, 2, 2, 3, 1, 4, 5, 3]

unique = []
seen = set()

for num in nums:
    if num not in seen:
        unique.append(num)
        seen.add(num)

print(unique)


#6. Write a program to count the number of words, lines, and characters in a text file.
filename = "sample.txt"

with open(filename, "r") as file:
    lines = file.readlines()

line_count = len(lines)
word_count = 0
char_count = 0

for line in lines:
    word_count += len(line.split())
    char_count += len(line)

print("Lines:", line_count)
print("Words:", word_count)
print("Characters:", char_count)

#7. Write a function to check whether two strings are anagrams (a word or phrase that is formed by arranging the letters of another word or phrase in a different order) of each other.
def are_anagrams(s1, s2):
    s1 = s1.replace(" ", "").lower()
    s2 = s2.replace(" ", "").lower()

    if len(s1) != len(s2):
        return False

    count = {}

    for ch in s1:
        count[ch] = count.get(ch, 0) + 1

    for ch in s2:
        if ch not in count or count[ch] == 0:
            return False
        count[ch] -= 1

    return True

print(are_anagrams("listen", "silent"))

#8. Write a program to remove punctuation from a given string.

text = input("Enter string: ")

punctuation = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/"

result = ""

for ch in text:
    if ch not in punctuation:
        result += ch

print(result)

#9. Write a program to find common elements between two lists.
list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

common = []

for item in list1:
    if item in list2 and item not in common:
        common.append(item)

print(common)

#10. Write a program to parse a simple JSON-like string into a dictionary without using the json module.
text = '{"name":"Isha","age":20,"city":"Delhi"}'

text = text.strip("{}")
pairs = text.split(",")

result = {}

for pair in pairs:
    key, value = pair.split(":")
    key = key.strip().strip('"')
    value = value.strip().strip('"')

    if value.isdigit():
        value = int(value)

    result[key] = value

print(result)

