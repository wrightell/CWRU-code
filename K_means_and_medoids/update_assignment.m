function assignment = update_assignment(means,data)
    p = length(data);
    assignment = 1:p;
    for i = 1:p
        [~,label] = min(vecnorm(means - data(:,i)));
        assignment(i) = label;
    end
end