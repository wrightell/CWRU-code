%this function creates a lattice and stores the points in a matrix
function lattice = create_lattice(size)
    x = size(1);
    y = size(2);
    lattice = zeros(2,x*y);
    counter = 1;
    for i = 1:x
        for j = 1:y
            lattice(:,counter) = [i,j];
            counter = counter + 1;
        end
    end
end