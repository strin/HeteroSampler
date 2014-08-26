function [] = vis(s)
    x = [];
    for i = 1:length(s)
        x = [x; s(i)-'0'];
    end
    x = reshape(x, 8, 16)';
    imshow(x, []);
end

