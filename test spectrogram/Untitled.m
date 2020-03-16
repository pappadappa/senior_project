result = [];

for i = 1:256
    if f1(i) == f2(i)
        result[i] = 1;
    else
        result[i] = 0;
    end
    
end
