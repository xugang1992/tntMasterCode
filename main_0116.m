%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the first collated version with annotation
% created on 02.12.2016 by Gang Xu

%-------------------------modification logs--------------------------------

% 1. Add a new Database called 'Beeler'
% 2. Add a objread function to read data from .obj file;
% 07.12.2016 by Gang Xu

%--------------------------------------------------------------------------
% 1. Add how to select levels and how to smooth
% 01.16.2017 by Gang Xu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
addpath(genpath('.'))

%% set parameters 
WHICH_DATABASE = 'Beeler';% bosphorus_database OR BU_database OR Beeler
USE_PREPARED_MASK = 1; 
ADD_MORE_POINTS   = 0;
DEBUG             = 0;
SCALE             = 6;
ADD_DETAILS       = 0;
USE_COARSE_MESH   = 0;
CONSIDER_ANGLE    = 0;
WEIGHT            = 0.06;


%% read data and generate Masks
switch WHICH_DATABASE
    case 'BU_database'
        disp('using BU database!');
        % M is the number of the person
        % N is the number of the image with the same expression
        % L is the string of the expression
        
        % M = 2+56;
        % N = 3; 
        % L = 5;

%         M = 84;
%         N = 4;
%         L = 7;

%         M = 22;
%         N = 3;
%         L = 7;

        M = 94;
        N = 1;
        L = 7;

%         M = 33;
%         N = 4;
%         L = 6;

%         M = 30;
%         N = 4;
%         L = 5;        
        % These strings are the abbreviation of expressions
        str={'NE','AN','DI','FE','HA','SA','SU'};
        if ~exist('data','var')
            load ('/home/xugang/Data_3D_Face/tmp/BU_3DFE_preprocessed.mat')            
        end        
        eval(['datai = data(M,N).' str{L} ';']);
        
        if USE_PREPARED_MASK
            eval(['load Mask_' num2str(M) '.mat'])
        else
            I = rgb2gray(imread(datai.imagename2D));
            figure()
            imshow(I,[])
            title('please select 2 points for cropping image');
            [x,y] = ginput(2);
            I = imcrop(I,[x(1),y(1),x(2)-x(1)+1,y(2)-y(1)+1]);
            ret = generateMasks(I,M);% This part can be generated with ffps.    
        end
        
%     case 'bosphorus_database'
%         disp('using bosphorus database!');
%         % M = 60;
%         % N = 1;
%         if ~exist('data','var')
%             load('/home/xugang/Data_3D_Face/tmp/bosphorus.mat')            
%         end
%         eval(['datai = data(M,N).' 'UFAU_2' ';']);
%         I = rgb2gray(imread(datai.image));
    case 'Beeler'
        disp('using Beeler database!');
        M = 7;
        if ~exist('datai','var')
            load('Beeler_18k_coarse_notRotated_without_outlier.mat')
            datai.connect = faces;
            datai.vertices = vertices;
        end
        if USE_PREPARED_MASK
            eval(['load Mask_' num2str(M) '.mat'])
        else             
            I = rgb2gray(imread('img_0003.jpg'));
            I = imresize(I,0.2);
%             figure()
%             imshow(I,[])
%             title('please select 2 points for cropping image');
%             [x,y] = ginput(2);
%             I = imcrop(I,[x(1),y(1),x(2)-x(1)+1,y(2)-y(1)+1]);        
            ret = generateMasks(I,M);% This part can be generated with ffps.    
        end    
    otherwise
        disp('can''t find the corresponding database!');
        return;
    
end

%% alignment and subdivision
% calculate the projection matrix, but now its quite simply caculated
% without considering the rotation

% step 1: delete NaN in input data
new_index_nan = isnan(datai.vertices);
new_index_nan = (sum(new_index_nan,2)~=0);
datai.vertices(new_index_nan,:)=[];


% Add more points if necessary.
if ADD_MORE_POINTS
    options.sub_type = 'loop';
    [newdata.vertices,newdata.faces] = perform_mesh_subdivision(datai.vertices,datai.connect,1,'loop');
else
    newdata.vertices = datai.vertices;
    newdata.faces = datai.connect;
end
if size(newdata.vertices,1) < size(newdata.vertices,2)
    newdata.vertices = newdata.vertices';
    newdata.faces = newdata.faces';
end

imgWidth = size(I,2);
imgHeight = size(I,1);

% figure()
% imshow(I,[])
% hold on 
% plot(imgWidth-x(datai.ffp(:,1))*imgWidth,imgHeight-y(datai.ffp(:,1))*imgHeight,'r.')
% title('check if the ffps are at the right position')


% Rotate the 3D mesh for better plotting.
switch WHICH_DATABASE
    case 'Beeler'
        ori.faces = newdata.faces;
        vertices2 = newdata.vertices;
        vertices2(:,2) = -vertices2(:,2);
        ori.vertices = vertices2(:,[3,2,1]);  
        plot_3D_mesh(ori);
%         title('This is the input 3D mesh')
    case 'BU_database'
        ori.faces = newdata.faces;
        ori.vertices = newdata.vertices;
        plot_3D_mesh(ori);
        title('This is the input 3D mesh')
end
% calculate the norm of each vertex
TR = triangulation(ori.faces,ori.vertices);
VN = vertexNormal(TR);
ErrorArray = zeros(2,100);
count = 1;
%% main part
for wnames={...
        'db4',...
        'db2','db20','haar',...
        'coif1','coif2','coif3',...        
        'sym2','sym3','sym4',...
       	'fk4', 'fk6', 'fk8', 'fk14', 'fk22',...
        'dmey',...
        'bior1.1', 'bior1.3', 'bior1.5',... 
        'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8',...
        'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7',...
        'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8',...
        'rbio1.1', 'rbio1.3', 'rbio1.5',... 
        'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8',...
        'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7',...
        'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'...
}
        wname = wnames{1};
%     close all
%     figure()
%     subplot(1,2,1)
%     imshow(I,[])
%     subplot(1,2,2)
%     imshow(I2,[])

%     I2 = imcrop(I,[76,267,95-76,281-267]);
%     I2 = imresize(I2,size(I));
%     I = medfilt2(I,[5,5]);
%     I = rgb2gray(imread('w2.JPG'));
%     I = imresize(I,[493,371]);
%     try    
%         w     = 5;       % bilateral filter half-width
%         sigma = [3 0.1]; % bilateral filter standard deviations
%         I = im2double(I);
%         I = bfilter2(I,w,sigma);
%         I = I*255;
%          
%             [reimage,diff] = BU_wavcoef_replace_finial(I,I2,scale,wname,debug);
%         diff= BU_wavcoef_replace_before(I,SCALE,wname,M,ADD_DETAILS,DEBUG);
        figure(),imshow(I)
    switch WHICH_DATABASE
        case 'Beeler'
        diff= BU_wavcoef_replace_M_after(I,SCALE,wname,M,WEIGHT,ADD_DETAILS,DEBUG);
        [xi,yi] = meshgrid(1:size(I,2),1:size(I,1));
        case 'BU_database'
        diff= BU_wavcoef_replace(I,SCALE,wname,M,WEIGHT,ADD_DETAILS,DEBUG);
        [xi,yi] = meshgrid(1:size(I,2),1:size(I,1));
            
    end
    switch WHICH_DATABASE
        case 'Beeler'
            cam = 'cam4.cam';       
            eval(['fid = fopen(''./cameraParemeters/' cam ''',''r'');']);

            if(fid == -1)
               fprintf('can not open the file');
            end

            tline = fgetl(fid);
            names = strsplit(tline,' ');
            tline= fgetl(fid);
            values = strsplit(tline,' ');
            fclose(fid);
            eval([names{1} '=''' values{1} ''';']);
            for i = 2:21
                eval([names{i} '= (' values{i} ');']);
            end    

            IntrinsicMatrix = [fx 0 cx; 0 fy cy; 0 0 1];

            Rx             = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
            Ry             = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
            Rz             = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
            RotationMatrix =  Rz*Ry*Rx;

            TranslationMatrix = [tx; ty; tz];
            ExtrinsicMatrix   = [RotationMatrix,TranslationMatrix];  
            Vnum = length(newdata.vertices(:,1));
            Hvertices = [newdata.vertices,ones(Vnum,1)];
            V2d = IntrinsicMatrix*ExtrinsicMatrix*Hvertices';
            V2d(1:2,:) = V2d(1:2,:)./repmat(V2d(3,:),2,1); 
            X = V2d(1,:)*0.2;
            Y = V2d(2,:)*0.2;
            offset = interp2(xi,yi,diff,X,Y,'cubic');
            offset = offset';
        case 'BU_database'
            MIN_X = min(newdata.vertices(:,1));
            x = newdata.vertices(:,1)-MIN_X;
            MAX_X = max(x);
            XI = x/MAX_X;

            MIN_Y = min(newdata.vertices(:,2));
            y = newdata.vertices(:,2)-MIN_Y;
            MAX_Y = max(y);
            YI = y/MAX_Y;
            offset = interp2(xi,yi,flip(diff,1),imgWidth*XI(:),imgHeight*YI(:),'cubic');
    end
         
        
%         wrong part, show for comparing
% % 
%         cam = 'cam5.cam';        
%         
%         eval(['fid = fopen(''./cameraParemeters/' cam ''',''r'');']);
% 
%         if(fid == -1)
%            fprintf('can not open the file');
%         end
%  
%         tline = fgetl(fid);
%         names = strsplit(tline,' ');
%         tline= fgetl(fid);
%         values = strsplit(tline,' ');
%         fclose(fid);
%         eval([names{1} '=''' values{1} '''']);
%         for i = 2:21
%             eval([names{i} '= (' values{i} ')']);
%         end    
%         fx = fx*0.9;
%         fy = fy*0.9;
%         IntrinsicMatrix = [fx 0 cx; 0 fy cy; 0 0 1];
% 
%         Rx             = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
%         Ry             = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
%         Rz             = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
%         RotationMatrix =  Rz*Ry*Rx;
%         tx = tx - 2;
%         ty = ty + 15;
%         TranslationMatrix = [tx; ty; tz];
%         ExtrinsicMatrix   = [RotationMatrix,TranslationMatrix];  
%         Vnum = length(newdata.vertices(:,1));
%         Hvertices = [newdata.vertices,ones(Vnum,1)];
%         V2d = IntrinsicMatrix*ExtrinsicMatrix*Hvertices';
%         V2d = V2d/mean(V2d(3,:)); 
%         X = V2d(1,:)*0.2;
%         Y = V2d(2,:)*0.2;
        
%         figure(10)
%         imshow(I)
%         hold on
%     
%         plot(X,Y,'r.','MarkerSize',1);
%         hold off
%         displacement(new_index_nan)=[];

%         figure()
%         surf(diff)
%         shading interp
%         colorbar

%         axis equal
    if CONSIDER_ANGLE
        angle    =  acos(sum(VN.*repmat([0,0,1],size(VN,1),1),2))*180/pi;               
%         thetaIdx =  abs(angle)<60;        
%         offset   =  VN.*repmat(offset,[1,3])./(repmat(180-angle,1,3)/180);
        offset   =  VN.*repmat(offset,[1,3]).*repmat(180-angle,1,3)/180;
%         offset   =  VN.*repmat(offset,[1,3])./cos(repmat(angle,1,3)/180*pi);
%         offset(~thetaIdx) = 0;
    else
        offset = VN.*repmat(offset,[1,3]);
    end

    newdata2.faces = ori.faces;
    newdata2.vertices = ori.vertices;      
    newdata2.vertices(:,1) = newdata2.vertices(:,1)+offset(:,1);
    newdata2.vertices(:,2) = newdata2.vertices(:,2)+offset(:,2);
    newdata2.vertices(:,3) = newdata2.vertices(:,3)+offset(:,3);


    rec.faces = newdata2.faces;
    rec.vertices = newdata2.vertices;
%         vertices2(:,2) = -vertices2(:,2);
%         rec.vertices = vertices2(:,[3,2,1]); 

    plot_3D_mesh(rec);
%     eval(['title(''Output, use ' wname ''')']);
    drawnow;
    saveas(gca,['./Results/3Dresults/' WHICH_DATABASE num2str(M) '_Result_scale' num2str(SCALE) '_' wname '_0010.eps'],'epsc');
  [ WHICH_DATABASE num2str(M) '_Result_scale' num2str(SCALE) '_' wname '_006.eps']
    figure() 
    imagesc(diff),colorbar,drawnow;
%       Checking if it's good aligned
    switch WHICH_DATABASE
        case 'Beeler'
            figure() 
            imshow(I,[]);
            hold on 
            plot(V2d(1,:)*0.2,V2d(2,:)*0.2,'r.','MarkerSize',0.1);
            hold off
         case 'BU_database'
            figure()
            imshow(flip(I,1),[]);
            hold on
            plot(imgWidth*XI(:),imgHeight*YI(:),'r.','MarkerSize',0.1);
            hold off
    end
%         plot_3D_mesh(newdata);
%         plot_face_model([new_x,new_y,new_z],new_tri,[49/255,165/255,234/255]);


    if strcmp(WHICH_DATABASE,'Beeler')
        load Beeler_18k_high_notRotated_without_outlier_0115.mat
        high.faces = faces;
        vertices2 = vertices;
        vertices2(:,2) = -vertices2(:,2);
        high.vertices = vertices2(:,[3,2,1]);  
        plot_3D_mesh(high);
%         title('This is the high fidelity 3d mesh')
    end

       if  strcmp(WHICH_DATABASE,'Beeler') 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % needed: shape_true, shape, tri
        % compare the input and output
        
        Prog =sum((high.vertices - ori.vertices).*VN,2);
        Sign = sign(Prog);
        diff = sqrt(sum((high.vertices - ori.vertices).^2,2));
        mean(diff)
        diff = Sign.*diff;        
        diff(diff>3)  = 3;
        diff(diff<-3) = -3;
%         region =  [5,16,-91,-63,140,164];
        region =  [6.7,12.7,-92,-76,144,158];
        index = high.vertices(:,1)>=region(1)&high.vertices(:,1)<=region(2)&...
            high.vertices(:,2)>=region(3)&high.vertices(:,2)<=region(4)&...
            high.vertices(:,3)>=region(5)&high.vertices(:,3)<=region(6);
        diff2 = sqrt(sum((high.vertices(index) - ori.vertices(index)).^2,2));        
        ErrorArray(1,count)=mean(diff2);
%         % sum over rows

%         diff_p = Matix(:);

        % paramters for plot
            facealpha = 1;
            ecol = 'none';
        % test on this
            cdatamapping = 'scaled';
        %     cdatamapping = 'direct';

        figure();
        patch('Faces',ori.faces,'Vertices',ori.vertices,'FaceVertexCData',diff,'FaceColor','interp',...
        'Edgecolor',ecol,'CDataMapping',cdatamapping,'FaceAlpha',facealpha,...
                    'FaceLighting','gouraud', 'AmbientStrength',0.5,...
        'SpecularStrength', 0.75);
                axis equal off tight
                lighting gouraud
            % select a unique color axis for all subplots.... (un)comment the
%         following...
%                 caxis manual
%                 caxis([0 13.*10^-3]);

                % shading interp
        %change light position
        %         light('Position',[-50 40 40],'Style','local')
        %         light('Position',[50 40 40],'Style','local')
                light('Position',[0 100 1000],'Style','local')
                colorbar;
        drawnow
        set(gcf,'color','w')
        

        
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %compare to the input
%         Code fragments:
        %%%%%%%%
        % needed: shape_true, shape, tri      

        Prog =sum((ori.vertices - rec.vertices).*VN,2);        
        Sign = sign(Prog);
        diff = sqrt(sum((ori.vertices - rec.vertices).^2,2));  
        mean(diff)
        diff = Sign.*diff;
        diff(diff>3)  = 3;
        diff(diff<-3) = -3;

        diff2 = sqrt(sum((ori.vertices(index) - rec.vertices(index)).^2,2));
        mean(diff2)


%         high_value = 2;
%         diff_p(diff_p >high_value) = high_value;

%         high_value = 2;
%         diff_p(diff_p >high_value) = high_value;
%         diff_p(diff_p>0.00001)=0; % testing

        % paramters for plot
        facealpha = 1;
        ecol = 'none';
        % test on this
        cdatamapping = 'scaled';
%             cdatamapping = 'direct';

        figure();
        patch('Faces',rec.faces,'Vertices',rec.vertices,'FaceVertexCData',diff,'FaceColor','interp',...
        'Edgecolor',ecol,'CDataMapping',cdatamapping,'FaceAlpha',facealpha,...
                    'FaceLighting','gouraud', 'AmbientStrength',0.5,...
        'SpecularStrength', 0.75);
        axis equal off tight
        lighting gouraud
            % select a unique color axis for all subplots.... (un)comment the
%         following...
%                 caxis manual
%                 caxis([0 13.*10^-3]);

                % shading interp
        %change light position
        %         light('Position',[-50 40 40],'Style','local')
        %         light('Position',[50 40 40],'Style','local')
%         light('Position',[0 100 1000],'Style','local')
        set(gcf,'color','w')
        colorbar;
        drawnow


        
        

%     catch
%         disp(['error for wname >',wname,'<. Skip it...'])
%         continue
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %compare to high
%         n_pts = size(high.vertices,1); % 3xn
  
        TR = triangulation(rec.faces,rec.vertices);
        VN = vertexNormal(TR);

        Prog =sum((high.vertices - rec.vertices).*VN,2);
        Sign = sign(Prog);
        diff = sqrt(sum((high.vertices - rec.vertices).^2,2));
        mean(diff)
        diff = Sign.*diff;       
        diff(diff>3) = 3;
        diff(diff<-3) = -3;
        

        diff2 = sqrt(sum((high.vertices(index) - rec.vertices(index)).^2,2));
        ErrorArray(2,count)=mean(diff2);

%         % sum over rows
%         diff_p = Matix(:);

        % paramters for plot
            facealpha = 1;
            ecol = 'none';
        % test on this
            cdatamapping = 'scaled';
        %     cdatamapping = 'direct';

        figure();
        patch('Faces',ori.faces,'Vertices',ori.vertices,'FaceVertexCData',diff,'FaceColor','interp',...
        'Edgecolor',ecol,'CDataMapping',cdatamapping,'FaceAlpha',facealpha,...
                    'FaceLighting','gouraud', 'AmbientStrength',0.5,...
        'SpecularStrength', 0.75);
                axis equal off tight
                lighting gouraud
            % select a unique color axis for all subplots.... (un)comment the
%         following...
%                 caxis manual
%                 caxis([0 13.*10^-3]);

                % shading interp
        %change light position
        %         light('Position',[-50 40 40],'Style','local')
        %         light('Position',[50 40 40],'Style','local')
                light('Position',[0 100 1000],'Style','local')
                colorbar;
        drawnow
        set(gcf,'color','w')
        count = count +1;
       end
       pause();
end


%%
figure()
plot(1:100,ErrorArray(1,:),'r-',1:100,ErrorArray(2,:),'b-');

v1 = ori.vertices;
v2 = high.vertices;

index = 200:300;
v1 = v1(index,:);
v2 = v2(index,:);

figure, 
plot3(v1(:,1),v1(:,2),v1(:,3),'r.','markersize',10)
hold on
plot3(v2(:,1),v2(:,2),v2(:,3),'g.','markersize',10)
plot3([v1(:,1) v2(:,1)]',[v1(:,2) v2(:,2)]',[v1(:,3) v2(:,3)]','k-')
axis equal



%%

% [R,T] = icp(ori.vertices,high.vertices,10);

