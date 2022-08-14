function centers = update_centers(D,labels,k)

        centers = (1:k);

        %udate each of the k medoids
        for i = 1:k

            %get indices of current medoid
            indices = find(labels == i);

            %get subdistance matrix of medoids
            medoid = D(indices,indices);

            %the new center is the one with the minimum coherence
            [~,new_center] = min(sum(medoid));

            %update the center of the medoid
            centers(i) = indices(new_center);

        end

end