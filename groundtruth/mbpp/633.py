"""
Write a python function to find the sum of xor of all pairs of numbers in the given list.
"""

def pair_xor_Sum(arr,n) : 
    ans = 0 
    for i in range(0,n) :    
        for j in range(i + 1,n) :   
            ans = ans + (arr[i] ^ arr[j])          
    return ans 



assert pair_xor_Sum([5,9,7,6],4) == 47
assert pair_xor_Sum([7,3,5],3) == 12
assert pair_xor_Sum([7,3],2) == 4