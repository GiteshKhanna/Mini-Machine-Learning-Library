fop  = open('C:\Python\CSV\data.csv','w')


fop.write('Sno.,Name,Year\n')
i  = 1
print("Press<enter> to stop inputting data")
while(1):
     fop.write(str(i))
     name = input("Name:")
     if(name is ''):
         break
     year = input("Year:")
     if(year is ''):
         fop.write(','+name)
         break
     fop.write(','+name+','+year+'\n')
     i = i+1


fop.close()

     


