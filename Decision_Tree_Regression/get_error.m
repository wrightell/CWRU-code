function error = get_error(split,data,vals)

        l = vals(data < split,:);
        r = vals(data >= split,:);
        
       meanl = mean(l,1);
       meanr = mean(r,1);

       ldiff = (l - meanl).^2;
       rdiff = (r - meanr).^2;

       lerr = 0;
       rerr = 0;
       for i = 1:3
            lerr = lerr + sum(ldiff(:,i));
            rerr = rerr + sum(rdiff(:,i));
       end

        error = lerr + rerr;
end