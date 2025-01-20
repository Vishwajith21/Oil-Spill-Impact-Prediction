"""Task 1 : Create a dataframe with 5 columns & 200 rows with random int  values 
--using that data , filter the first column -greater than 50
--Statistics About the Data"""


def hjhj(a):
    l=len(a)
    dupeli=[]
    for i in range (l):
        lk= i+1
        for j in range (lk,l):
            if a[i]==a[j] and a[i] not in dupeli:
                dupeli=dupeli+[a[i]]
    return dupeli

print(hjhj([1,2,3,1]))