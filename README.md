## Алгоритмы и структуры данных

### Списки

Списки - это тип структуры данных, который позволяет хранить разные значения.  
Списки связывают данные с помощью указателей. Указатели указывают на следующую 
порцию данных.

В списках данные хранятся в различных, не связанных между собой областях памяти

<img src="media_readme/list_memory.jpg" alt="list_memory" width="350"/>

Так как данные располагаются в различных местах памяти, то доступ может быть 
только через указатели

### Массивы

Массивы - это тип структуры данных, позволяющий хранить несколько значений.  
Каждый элемент доступен через его индекс, который означает положение элемента
в массиве. Данные хранятся в памяти последовательно

<img src="media_readme/array_memory.jpg" alt="array_memory" width="350"/>

Так как данные располагаются последовательно, адреса в памяти вычисляются с помощью индексов, 
позволяя организовать произвольный доступ к данным. Но массивы имеют свои минусы, такие как
высокая стоимость добавления или удаления данных по сравнению со списками

### Хеш-таблицы

Хеш-таблицы - это разновидность структур данных, подходящие для хранения данных в наборах, 
состоящих из **ключей** и **значений**

### Задачи

<details>
<summary><b>Arrays and hashing:</b></summary>  
<br>

<details>
<summary><b>217. Contains-duplicate:</b></summary>

https://leetcode.com/problems/contains-duplicate/
<img src="media_readme/leetcode_tasks/array_and_hashing/217.png"/>

Time: O(nlog(n)); Space: O(1)
```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums = sorted(nums)

        for n in range(len(nums) - 1):
            if nums[n] == nums[n+1]:
                return True
        return False
```

Time: O(n); Space: O(n)
```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashset = set ()

        for n in nums:
            if n in hashset:
                return True
            
            hashset.add(n)
        return False
```
</details>

<details>
<summary><b>242. Valid Anagram</b></summary>
<img src="media_readme/leetcode_tasks/array_and_hashing/217.png"/>

Time: O(n); Space: O(n);
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        '''
        Пройдемся по каждому символу в строке, каждый уникальный символ будем добавлять в качестве ключа,
        если символ уже находится в словаре, то увеличиваем счетчик
        Выполняем для двух строк
        Сравниваем словари
        '''
        if len(s) != len(t):
            return False
        
        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)            
        
        for c in countS:
            if countS[c] != countT.get(c, 0):
                return False
        
        return True
```

Time: O(nlog(n)); Space: O(nlog(n));
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)
        
```
</details>

<details>
<summary><b>1. Two Sum</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/1.png" />

Time: O(n); Space: O(n);
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        '''
        Создадим словарь с парой число:индекс_числа
        Проиндекстируем список nums
        Найдем разницу между target и числом
        Если число размером в разницу есть в словаре, то возвращаем результат
        Иначе добавляем число в словарь   
        '''
        hashmap = {}  # value : index

        for k, v in enumerate(nums):
            diff = target - v
            if diff in hashmap:
                return [hashmap[diff], k]
            hashmap[v] = k
```
</details>

<details>
<summary><b>49. Group Anagrams</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/49.png" />

Time: O(n * k * log(k)); Space: O(n);
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagram_dict = defaultdict(list)
        
        for i, word in enumerate(strs):
            sorted_word = ''.join(sorted(word))
            anagram_dict[sorted_word].append(word)

        return list(anagram_dict.values())
```
</details>

<details>
<summary><b>347. Top K Frequent Elements</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/347.png" />

Time: O(n * k * log(k)); Space: O(n);
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Используем Counter для подсчета частоты элементов
        num_count = Counter(nums)

        # Сортируем элементы по частоте в убывающем порядке
        sorted_nums = sorted(num_count, key=lambda x: num_count[x], reverse=True)

        # Возвращаем первые k элементов
        return sorted_nums[:k]
```
</details>

<details>
<summary><b>605. Can Place Flowers</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/605.png" />

Time: O(n); Space: O(1);
```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        length = len(flowerbed)
        i = 0

        while i < length:
            if flowerbed[i] == 0:
                if i == length - 1 or flowerbed[i + 1] == 0:
                    n -= 1
                    i += 2  # пропускаем два элемента, так как они не могут влиять на посадку цветка
                else:
                    i += 3  # пропускаем три элемента, так как следующий элемент занят
            else:
                i += 2  # пропускаем два элемента, так как текущий элемент уже занят

            if n <= 0:
                return True

        return False
```
</details>

<details>
<summary><b>941. Valid Mountain Array</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/941.png" />

Time: O(n); Space: O(1);
```python
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        l = len(arr)

        if l < 3:
            return False
        
        i = 0
        while arr[i] < arr[i+1]:
            i += 1 
            if i == l - 1:
                return False  
        
        j = l-1
        while arr[j-1] > arr[j]:
            j -= 1
            if j == 0:
                return False 
        
        return i == j
```
</details>

<details>
<summary><b>228. Summary Ranges</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/228.png" />

Time: O(n); Space: O(1);
```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        i = 0
        j = 0
        res = []

        while j <= len(nums) - 1:
            start = nums[i]

            while j < len(nums) - 1 and nums[j + 1] - nums[j] == 1:
                j += 1

            finish = nums[j]

            if start == finish:
                res.append(str(start))
            else:
                res.append(str(start) + "->" + str(finish))

            i = j + 1
            j += 1
        return res
```
</details>

<details>
<summary><b>1431. Kids With the Greatest Number of Candies</b></summary>
<img src="media_readme/leetcode_tasks/array_and_hashing/1431.png" />

Time: O(n); Space: O(n);
```python
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        return [candy + extraCandies >= max(candies) for candy in candies]
```
</details>

<details>
<summary><b>674. Longest Continuous Increasing Subsequence</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/674.png" />

Time: O(n); Space: O(n);

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if not nums:
            return 0

        res = [1]
        f = 1

        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                res.append(res[-1] + 1)
            else:
                res.append(1)

        return max(res)
```
</details>

<details>
<summary><b>1480. Running Sum of 1d Array</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/1480.png" />

Time: O(n); Space: O(1);

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i-1]
        return nums
```
</details>

<details>
<summary><b>896. Monotonic Array</b></summary>

<img src="media_readme/leetcode_tasks/array_and_hashing/896.png" />

Time: O(n); Space: O(1);

```python
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        i = 0
        while i < len(nums) - 1 and nums[i] <= nums[i+1]:
            i += 1
        
        j = 0
        while j < len(nums) - 1 and nums[j] >= nums[j+1]:
            j += 1
        
        return any([i == len(nums) - 1, j == len(nums) - 1])
```
</details>

</details>

<details>
<summary><b>Binary Search:</b></summary>
<br>
<details>
<summary><b>367. Valid Perfect Square</b></summary>

<img src="media_readme/leetcode_tasks/divmod/367.png" />

Time: O(1); Space: O(1);

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        return num % num ** 0.5 == 0
```

Time: O(log(n)); Space: O(1);
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        l, r = 1, num // 2

        if num == 1:
            return True

        while l <= r:
            mid = (l + r ) // 2
            
            if mid ** 2 == num:
                return True
            
            if mid ** 2 > num:
                r = mid - 1
            else:
                l = mid + 1
        
        return False
```
</details>

<details>
<summary><b>704. Binary Search</b></summary>

<img src="media_readme/leetcode_tasks/binary search/704.png" />

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (r + l) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid

        return -1
```
</details>

<details>
<summary><b>69. Sqrt(x)</b></summary>

<img src="media_readme/leetcode_tasks/binary search/69.png" />

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r = 1, x

        while l <= r:
            mid = (l + r) // 2

            if mid * mid == x:
                return int(mid)
            
            if mid * mid < x:
                l = mid + 1
            else:
                r = mid - 1

        return r
```
</details>

<details>
<summary><b>278. First Bad Version</b></summary>

<img src="media_readme/leetcode_tasks/binary search/278.png" />

Time: O(log(n)); Space: O(1);

```python
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        l, r = 0, n
        
        while l <= r:
            mid = (l + r) // 2

            if isBadVersion(mid) == False:
                l = mid + 1
            else:
                r = mid - 1
            
            if isBadVersion(mid-1) == False and isBadVersion(mid) == True:
                return mid
```
</details>

<details>
<summary><b>374. Guess Number Higher or Lower</b></summary>

<img src="media_readme/leetcode_tasks/binary search/374.png" />

Time: O(log(n)); Space: O(1);

```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        l, r = 1, n

        while l <= r:
            mid = (l + r) // 2
            g = guess(mid)
            
            if g == 0:
                return mid
            elif g > 0:
                l = mid + 1
            else:
                r = mid - 1

        return r
```
</details>

<details>
<summary><b>35. Search Insert Position</b></summary>

<img src="media_readme/leetcode_tasks/binary search/35.png" />

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        if len(nums) == 1 and nums[0] < target:
            return 1

        while l < r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid

            if nums[-1] < target:
                return len(nums)
            elif nums[0] > target:
                return 0

        return r
```
</details>

<details>
<summary><b>852. Peak Index in a Mountain Array</b></summary>

<img src="media_readme/leetcode_tasks/binary search/852.png" />

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        l, r = 0, len(arr) - 1

        while l <= r:
            mid = (l + r) // 2

            if arr[mid - 1] < arr[mid] > arr[mid + 1]:
                return mid

            if arr[mid - 1] < arr[mid]:
                l = mid + 1
            else:
                r = mid
```
</details>

</details>

<details>
<summary><b>Bitwise</b></summary><br>
<br>
<details>
<summary><b>338. Counting Bits</b></summary>

<img src="media_readme/leetcode_tasks/bitwise/338.png"/>

Time: O(n * log(n)); Space: O(1);

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        # ans = [bin(i)[2:].count('1') for i in range(n+1)]

        ans = [0]

        for i in range(1, n + 1):
            current = 0

            while i:
                current += i & 1
                i >>= 1
            ans.append(current)

        return ans
```
</details>

<details>
<summary><b>136. Single Number</b></summary>

<img src="media_readme/leetcode_tasks/bitwise/136.png" />

Time: O(n); Space: O(1);

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        
        for num in nums:
            ans = ans ^ num
        
        return ans
```
</details>

<details>
<summary><b>461. Hamming Distance</b></summary>

<img src="media_readme/leetcode_tasks/bitwise/461.png" />

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:

        ans = 0

        while x or y:

            ans += (x & 1) != (y & 1)
            x >>= 1
            y >>= 1

        return ans
```
</details>

<details>
<summary><b>191. Number of 1 Bits</b></summary>

<img src="media_readme/leetcode_tasks/bitwise/191.png" />

Time: O(n); Space: O(1);

```python
class Solution:
    def hammingWeight(self, n: int) -> int:

        count = 0

        while n:
            
            count += n & 1
            n >>= 1
        
        return count
```
</details>

</details>

<details>
<summary><b>Divmod</b></summary><br>
<br>

<details>
<summary><b>9. Palindrome Number</b></summary>

<img src="media_readme/leetcode_tasks/divmod/9.png"/>

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        num = 0
        orig = x

        while x:
            x, digit = divmod(x, 10)  # Берем последнюю цифру числа

            num = num * 10 + digit  # Пересобираем число справа налево
        
        return num == orig
```
</details>

<details>
<summary><b>258. Add Digits</b></summary>

<img src="media_readme/leetcode_tasks/divmod/258.png"/>

Time: O(log(n)); Space: O(1);

```python
class Solution:
    def addDigits(self, num: int) -> int:
        new = 0

        while num:
            num, digit = divmod(num, 10)

            new = new + digit

        if new > 9:
            return self.addDigits(new)

        return new
```
</details>

<details>
<summary><b>66. Plus One</b></summary>

<img src="media_readme/leetcode_tasks/divmod/66.png"/>

Time: O(n); Space: O(1);

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1

        for i in range(len(digits)-1, -1, -1):

            carry, digits[i] = divmod(carry+digits[i], 10)

            if carry == 0:
                break

        return digits if not carry else [carry] + digits
```
</details>

<details>
<summary><b></b></summary><br>
<br>

<details>
<summary><b></b></summary>

<img src=""/>

Time: O(); Space: O();

```python

```
</details>

</details>























<details>
<summary><b></b></summary><br>
<br>

<details>
<summary><b></b></summary>

<img src=""/>

Time: O(); Space: O();

```python

```
</details>

</details>






###############################

<details>
<summary><b></b></summary><br>
<br>

<details>
<summary><b></b></summary>

<img src=""/>

Time: O(); Space: O();

```python

```
</details>

</details>





