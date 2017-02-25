function diff=BU_wavcoef_replace_M_after(I,scale,wavename,M,weight,ADD_DETAILS,debug)
    

%      weight1 = 0.05;
%     weight2 = -0.3;
    weight3 = 0.8;
    

    for y =1
        tic
        I2 = I;
        eval(['load Mask_' num2str(M) '.mat'])
%         Mask_background = 1-Mask_forehead;        
%         I2(logical(Mask_background))= I2(100,100);
%         figure(),imshow(I2,[]);
        [C,S] = wavedec2(I2,scale,wavename);
        if debug
            ind = 1;
            CA = C(ind:S(1,1)*S(1,2));
            CA = (CA - min(CA(:)))/(max(CA(:)-min(CA(:))));
            CA = reshape(CA,S(1,1),S(1,2));
            ind = ind + S(1,1)*S(1,2);

            for i = 2:scale+1

               CH = C(ind:ind+S(i,1)*S(i,2)-1);
               CH = (CH - min(CH(:)))/(max(CH(:)-min(CH(:))));
               CH = reshape(CH,S(i,1),S(i,2));
               ind = ind + S(i,1)*S(i,2);

               CV = C(ind:ind+S(i,1)*S(i,2)-1);   
               CV = (CV - min(CV(:)))/(max(CV(:)-min(CV(:))));
               CV = reshape(CV,S(i,1),S(i,2));
               ind = ind + S(i,1)*S(i,2);


               CD = C(ind:ind+S(i,1)*S(i,2)-1);
               CD = (CD - min(CD(:)))/(max(CD(:)-min(CD(:))));
               CD = reshape(CD,S(i,1),S(i,2));
               ind = ind + S(i,1)*S(i,2);
               if size(CA,1)~=size(CH,1) || size(CA,2)~=size(CH,2)
                   CA = imresize(CA,size(CH));
               end
               CA = [CA,CH;CV,CD];
            end
            figure(),imshow(CA,[]);title('Reconstructed map coefficients');
        end
        
                        
        C_new = zeros(size(C));
        Mask = Mask_face - Mask_nose - Mask_left_eyebrow -...
            Mask_right_eyebrow - Mask_left_eye_ball -...
            Mask_right_eye_ball - Mask_mouth;       
        Mask = imgaussfilt(double(Mask),5);

%         figure();imshow(Mask,[]);

        ind = 1;
        for i = 1:scale+1
            if i==1
%                  C(ind:S(i,1)*S(i,2)) = 0;
                 ind = ind + S(i,1)*S(i,2);
            elseif i==scale+1
                C_new(ind:end) = 0;
%             elseif i==scale
%                 C_new(ind:end) = 0;
%             elseif i==scale
%                 C_new(ind:end) = 0;

            else
                for j = 1:3
                    C_new(ind:ind+S(i,1)*S(i,2)-1) = C_new(ind:ind+S(i,1)*S(i,2)-1) +...
                        weight*C(ind:ind+S(i,1)*S(i,2)-1);
                    ind = ind + S(i,1)*S(i,2);
                end
            end   
        end
        if ADD_DETAILS                        
            ind = 1;
            C_new2 = zeros(size(C));
            for i = 1:scale+1
                if i==1
        %                  C(ind:S(i,1)*S(i,2)) = 0;
                     ind = ind + S(i,1)*S(i,2);
                elseif i==scale+1
                    C_new2(ind:end) = weight3*C(ind:end);
        %             elseif i==scale
        %                 C_new(ind:end) = 0;
        %             elseif i==scale
        %                 C_new(ind:end) = 0;

                else
                    for j = 1:3
%                         C_new(ind:ind+S(i,1)*S(i,2)-1) = C_new(ind:ind+S(i,1)*S(i,2)-1) +...
%                             weight1*C(ind:ind+S(i,1)*S(i,2)-1);
                        ind = ind + S(i,1)*S(i,2);
                    end
                end   
            end
        end   
        
%         C_new = C_new + C_new2;
        THR = thselect(C_new,'rigrsure');
        C_new = wthresh(C_new,'s',THR);
        displacement_mainpart = waverec2(C_new,S,wavename);
        
        if ADD_DETAILS
            THR = thselect(C_new2,'rigrsure');
            C_new2 = wthresh(C_new2,'s',THR);
            displacement_mainpart2 = waverec2(C_new2,S,wavename);       
            displacement_mainpart = (displacement_mainpart+displacement_mainpart2).*Mask;
        else
            displacement_mainpart = displacement_mainpart.*Mask;
        end
        
        g = fspecial('gaussian', 5,2);
        displacement_mainpart = imfilter(displacement_mainpart,g,'replicate');

        displacement_mainpart = medfilt2(displacement_mainpart,[3,3]);
        
        diff = displacement_mainpart;

    end


end


