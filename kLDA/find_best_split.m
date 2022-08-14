function split = find_best_split(p,I)
    m1 = mean(p(I == -1));
    m2 = mean(p(I == 1));
    t = linspace(m1,m2);
    split = 0;
    split_acc = 0;
    hundreds = [];
    for i = 1:length(t)
        acc = sum(sign(p - t(i)) == I)/length(p);
        if acc == 100
            hundreds = [hundreds, t(i)];
        end
        if acc > split_acc
            split_acc = acc;
            split = t(i);
        end
    end
    
    if ~isempty(hundreds)
        split = mean(hundreds);
    end


end