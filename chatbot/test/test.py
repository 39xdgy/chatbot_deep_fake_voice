tell_chloe = "hi i am vicki" 
chloes_reply = talk_to_chloe(tell_chloe, chloe, opt, infield, outfield)
print('Chloe > '+ chloes_reply + '\n')

while True:
    tell_chloe = input("You > ")
    chloes_reply = talk_to_chloe(tell_chloe, chloe, opt, infield, outfield)
    if ("bye chloe" in tell_chloe or "bye ttyl" in chloes_reply):
        print('Chloe > '+ chloes_reply + '\n')
        break
    else:
        print('Chloe > '+ chloes_reply + '\n') 