function [M,Is] = LVQ(data,I,k)
    
    %get number of classes
    num_classes = length(unique(I));

    %initizalize set of all prototypes
    M = rand(length(data(:,1)),k*num_classes);
    Is = zeros(1,k*num_classes);
    
    %for each class generate prototypes
    for i = 1:num_classes

        subs = data(:,I == i-1);
        M(:,(i-1)*k+1:i*k) = subs(:,randsample(length(subs(1,:)),k));
        Is((i-1)*k+1:i*k) = i-1;
    end
    

    %move each prototype
    for t = 1:1000
        p = randi(length(I));
        a = .9*exp(-t*log(10)/1000);
        point = data(:,p);
        [~,index] = min(vecnorm(M-point));
        if(Is(index) == I(p))
           M(:,index) = M(:,index) + a*(point-M(:,index));
        else
            M(:,index) = M(:,index) - a*(point-M(:,index));
        end
        
    end

end