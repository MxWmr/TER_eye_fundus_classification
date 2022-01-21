using Images, FileIO
using Statistics

function reduction(img)
    im=Gray.(img)
    # 2048 =64x32  3072=64x48  we start with big squares for less calculus and it's working well
    max=0.0  #we search for the most luminous square
    for i in 6:26
        for j in 12:36
            im_cop=im[(64*(i-1))+1:64*i,(64*(j-1))+1:64*j]
            m=mean(im_cop)
            if m>max
                max=m
                global ic=i
                global jc=j
            end
        end
    end
    im_reduc = img[ic*64-316:ic*64+284,jc*64-316:jc*64+284]
    return im_reduc
end