import argparse
if __name__ =="__main__" :
    parser = argparse.ArgumentParser(description= "calculator App")
    parser.add_argument("--number1",help = "first number")
    parser.add_argument("--number2",help = "2nd number")
    parser.add_argument("--operation",help = "Operation",\
                        choices =["add","sub","mul"])

    arg = parser.parse_args()

    print(arg.number1) 
    print(arg.number2) 
    print(arg.operation) 

    n1,n2 = int(arg.number1),int(arg.number2)
    result = None
    if arg.operation =="add":
        result = n1+n2
    elif arg.operation =="sub":
        result = n1-n2
    elif arg.operation =="mul":
        result = n1*n2
    else :
        print("unsupported operation")    

    print("result ::", result)