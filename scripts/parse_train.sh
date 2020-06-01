cat train128.txt | grep "Step =" | tr -s ' ' | cut -d" " -f3,6,17,21,24 
