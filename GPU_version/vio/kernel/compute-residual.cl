// Copyright (C) 2021  Majid Geravand
// Copyright (C) 2021  Gfuse

// Enable OpenCL 32-bit integer atomic functions.
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

float2 world2cam(float3 feature)
{
    float r = sqrt(pow(feature.x/feature.z, 2) + pow(feature.y/feature.z, 2));
    float factor = 1.0;
    if((float)S == 0 || r < 0.001){
        factor = 1.0;
    }else{
        float y=2*r*sin(0.5 * (float)S);
        float x=cos(0.5 * (float)S);
        factor = (atan2(y,x) / (r * (float)S));
    }
    return (float2)((float)C_X + (float)F_X * factor * feature.x/feature.z, (float)C_Y + (float)F_Y * factor * feature.y/feature.z);
}

float3 xyz_cur(float3 cur, float3 ref, float3 ref_feature)
{
    ref.x*=-1.0;
    ref.y*=-1.0;
    ref.z = 3.141592653589793238462643383279502884197169399375 + ref.z;
    float3 error=ref+cur;
    float yaw=0.0;
    float pitch=error.z;
    float roll=0.122173;
    float R00=cos(yaw)*cos(pitch);//cos(pitch)
    float R01=cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll);//0
    float R02=cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll);//sin(pitch)
    float R10=sin(yaw)*cos(pitch);//0
    float R11=sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll);//1
    float R12=sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll);//0
    float R20=-sin(pitch);//-sin(pitch)
    float R21=cos(pitch)*sin(roll);//0
    float R22=cos(pitch)*cos(roll);//cos(pitch)
    return  (float3)(R00*ref_feature.x+R01*ref_feature.y+R02*ref_feature.z+error.x,
                     R10*ref_feature.x+R11*ref_feature.y+R12*ref_feature.z,
                     R20*ref_feature.x+R21*ref_feature.y+R22*ref_feature.z+error.y);
}

void jacobian_xyz2uv(float3 xyz_in_f, float* J)
{
    J[0] = -(1. / xyz_in_f.z);                                         // -1/z
    J[1] = (xyz_in_f.x) / (xyz_in_f.z * xyz_in_f.z);                   // x/z^2
    J[2] = -(1.0 + pow((xyz_in_f.x),2) / (xyz_in_f.z * xyz_in_f.z));   // -(1.0 + x^2/z^2)
    J[3] = 1e-19;                                                      // 0
    J[4] = (xyz_in_f.y) / (xyz_in_f.z * xyz_in_f.z);                   // y/z^2
    J[5] = -(xyz_in_f.x) * (xyz_in_f.y) / (xyz_in_f.z * xyz_in_f.z);   // -x*y/z^2
}

void jacobian_xyz2uv_(float3 xyz_in_f, float3 cur_p, float* J)
{
    double x_n = xyz_in_f.x;
    double y_n = xyz_in_f.y;
    double z_n = xyz_in_f.z;

    double r = sqrt(pow(xyz_in_f.x/xyz_in_f.z, 2) + pow(xyz_in_f.y/xyz_in_f.z, 2));

    double x_c = cur_p.x;
    double z_c = cur_p.y;
    double theta = cur_p.z;

    double alpha = (F_X*(theta/r))-(F_X*((x_n*x_n)/(r*r))*theta)+((1+3*S*theta*theta)/((r*r)+1))*((F_X*x_n*x_n)/(r*r));
    double beta  =                -(F_X*((x_n*y_n)/(r*r))*theta)+((1+3*S*theta*theta)/((r*r)+1))*((F_X*x_n*y_n)/(r*r));
    double gamma =                -(F_Y*((x_n*y_n)/(r*r))*theta)+((1+3*S*theta*theta)/((r*r)+1))*((F_Y*x_n*y_n)/(r*r));
    double lamda = (F_Y*(theta/r))-(F_Y*((y_n*y_n)/(r*r))*theta)+((1+3*S*theta*theta)/((r*r)+1))*((F_Y*y_n*y_n)/(r*r));

    double Xf_Xc = x_n - x_c;
    double Zf_Zc = z_n - z_c;
    double n1 = -1*sin(theta)*Xf_Xc + cos(theta)*Zf_Zc;
    double n2 = -1*cos(theta)*Xf_Xc - sin(theta)*Zf_Zc;

    J[0] = ((-1*cos(theta)/z_n)*alpha)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*sin(theta));
    J[1] = ((-1*sin(theta)/z_n)*alpha)+(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*cos(theta));
    J[2] = ((1/z_n)*alpha*n1)-(((x_n/(z_n*z_n))*alpha + (y_n/(z_n*z_n))*beta)*n2);

    J[3] = ((-1*cos(theta)/z_n)*gamma)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*sin(theta));
    J[4] = ((-1*sin(theta)/z_n)*gamma)+(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*cos(theta));
    J[5] = ((1/z_n)*gamma*n1)-(((x_n/(z_n*z_n))*gamma + (y_n/(z_n*z_n))*lamda)*n2);
}

void compute_hessain(float3 j, __global float* H, float w)
{
    H[0] += j.x * j.x * w;
    H[1] += j.x * j.y * w;
    H[2] += j.x * j.z * w;
    H[3] += j.y * j.x * w;
    H[4] += j.y * j.y * w;
    H[5] += j.y * j.z * w;
    H[6] += j.z * j.x * w;
    H[7] += j.z * j.y * w;
    H[8] += j.z * j.z * w;
}

__kernel void compute_residual(
        __read_only  image2d_t   image_cur, // current frame
        __read_only  image2d_t   image_ref, // reference frame
        __global     float3      * cur_pose,//[reference frame pose{x,z,pitch}]
        __global     float3      * ref_pose,//[current frame pose{x,z,pitch}]
        __global     float3      * ref_feature, // feature on the reference frame, when we applied the distance calculation: xyz_ref((*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0],0.0,ref_pos[1])).norm());
        __global     float2      * featue_px,
                     int           level,
        __global     float       * errors,
        __global     float       * Hessian,
        __global     float3      * Jacobian,
        __global     float       * chi,
                     float         scale_
)
{
float scale = pow(2.0, -level);
// Prepare a suitable OpenCL image sampler.
sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// Use global work feature.
int f = get_global_id(0);
// check if reference with patch size is within image
float2 uv_ref = featue_px[f] * scale;
float2 uv_ref_i = floor(uv_ref);
if(uv_ref_i.x - (PATCH_HALFSIZE + 1) < 0 || uv_ref_i.y - (PATCH_HALFSIZE + 1) < 0 || uv_ref_i.x + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref).x || uv_ref_i.y + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref).y)return;
// evaluate projection jacobian
float frame_jac[6]={0.0}; // 2X3
//jacobian_xyz2uv(sqrt(pow(ref_feature[f] - (float3)(ref_pose[0].x, 0.0, ref_pose[0].y), 2.0)), &frame_jac);
jacobian_xyz2uv_(ref_feature[f], cur_pose[0], &frame_jac);

// compute bilateral interpolation weights for reference image
float2 subpix=  uv_ref - uv_ref_i;
float w_ref_tl = (1.0 - subpix.x) * (1.0 - subpix.y);
float w_ref_tr = subpix.x * (1.0 - subpix.y);
float w_ref_bl = (1.0 - subpix.x) * subpix.y;
float w_ref_br = subpix.x * subpix.y;

// compute bilateral interpolation weights for the current image
float2 uv_cur_pyr = world2cam(xyz_cur(cur_pose[0], ref_pose[0] ,ref_feature[f])) * scale;
float2 uv_cur_i = floor(uv_cur_pyr);
float2 subpix_uv_cur = uv_cur_pyr - uv_cur_i;
float w_cur_tl = (1.0 - subpix_uv_cur.x) * (1.0 - subpix_uv_cur.y);
float w_cur_tr = subpix_uv_cur.x * (1.0 - subpix_uv_cur.y);
float w_cur_bl = (1.0 - subpix_uv_cur.x) * subpix_uv_cur.y;
float w_cur_br = subpix_uv_cur.x * subpix_uv_cur.y;
float e = 0.0;
float chi_=0.0;
for(int y = 0; y < PATCH_SIZE; ++y)
{
int ref_element_addr = (int)(uv_ref_i.y + y - PATCH_HALFSIZE) * get_image_dim(image_ref).x + (uv_ref_i.x - PATCH_HALFSIZE);
int cur_element_addr = (int)(uv_cur_i.y + y - PATCH_HALFSIZE) * get_image_dim(image_cur).x + (uv_cur_i.x - PATCH_HALFSIZE);

for(int x = 0; x < PATCH_SIZE; ++x, ++ref_element_addr, ++cur_element_addr)
{
// precompute interpolated reference patch color
int2 px_reftl = (int2)( ref_element_addr                                    % get_image_dim(image_ref).x,  ref_element_addr                                    / get_image_dim(image_ref).x);
int2 px_reftr = (int2)((ref_element_addr + 1)                               % get_image_dim(image_ref).x, (ref_element_addr + 1)                               / get_image_dim(image_ref).x);
int2 px_refbl = (int2)((ref_element_addr + get_image_dim(image_ref).x)     % get_image_dim(image_ref).x, (ref_element_addr + get_image_dim(image_ref).x)     / get_image_dim(image_ref).x);
int2 px_refbr = (int2)((ref_element_addr + get_image_dim(image_ref).x + 1) % get_image_dim(image_ref).x, (ref_element_addr + get_image_dim(image_ref).x + 1) / get_image_dim(image_ref).x);
float value = w_ref_tl * read_imageui(image_ref, sampler, px_reftl).x + w_ref_tr * read_imageui(image_ref, sampler, px_reftr).x +
              w_ref_bl * read_imageui(image_ref, sampler, px_refbl).x + w_ref_br * read_imageui(image_ref, sampler, px_refbr).x;

float dx = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 1)                                     % get_image_dim(image_ref).x, ((ref_element_addr + 1)                                     / get_image_dim(image_ref).x))).x +
                    w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 2)                                     % get_image_dim(image_ref).x, ((ref_element_addr + 2)                                     / get_image_dim(image_ref).x))).x +
                    w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) + 1)     % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) + 1)     / get_image_dim(image_ref).x))).x +
                    w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) + 2)     % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) + 2)     / get_image_dim(image_ref).x))).x)
                   -(w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr - 1)                                     % get_image_dim(image_ref).x, ((ref_element_addr - 1)                                     / get_image_dim(image_ref).x))).x +
                     w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 0)                                     % get_image_dim(image_ref).x, ((ref_element_addr + 0)                                     / get_image_dim(image_ref).x))).x +
                     w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) - 1)     % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) - 1)     / get_image_dim(image_ref).x))).x +
                     w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x))         % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x))         / get_image_dim(image_ref).x))).x));

float dy = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x))         % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x))         / get_image_dim(image_ref).x))).x +
                    w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) + 1)     % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) + 1)     / get_image_dim(image_ref).x))).x +
                    w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) * 2)     % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) * 2)     / get_image_dim(image_ref).x))).x +
                    w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref).x) * 2 + 1) % get_image_dim(image_ref).x, ((ref_element_addr + (get_image_dim(image_ref).x) * 2 + 1) / get_image_dim(image_ref).x))).x)
                   -(w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (-get_image_dim(image_ref).x))        % get_image_dim(image_ref).x, ((ref_element_addr + (-get_image_dim(image_ref).x))        / get_image_dim(image_ref).x))).x +
                     w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (1 - get_image_dim(image_ref).x))     % get_image_dim(image_ref).x, ((ref_element_addr + (1 - get_image_dim(image_ref).x))     / get_image_dim(image_ref).x))).x +
                     w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 0)                                     % get_image_dim(image_ref).x, ((ref_element_addr + 0)                                     / get_image_dim(image_ref).x))).x +
                     w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 1)                                     % get_image_dim(image_ref).x, ((ref_element_addr + 1)                                     / get_image_dim(image_ref).x))).x));

// compute residual
int2 px_curtl = (int2)( cur_element_addr                                      % get_image_dim(image_cur).x,  cur_element_addr                                      / get_image_dim(image_cur).x);
int2 px_curtr = (int2)((cur_element_addr + 1)                                 % get_image_dim(image_cur).x, (cur_element_addr + 1)                                 / get_image_dim(image_cur).x);
int2 px_curbl = (int2)((cur_element_addr + (get_image_dim(image_cur).x))     % get_image_dim(image_cur).x, (cur_element_addr + (get_image_dim(image_cur).x))     / get_image_dim(image_cur).x);
int2 px_curbr = (int2)((cur_element_addr + (get_image_dim(image_cur).x) + 1) % get_image_dim(image_cur).x, (cur_element_addr + (get_image_dim(image_cur).x) + 1) / get_image_dim(image_cur).x);

float res = value - w_cur_tl * read_imageui(image_cur, sampler, px_curtl).x + w_cur_tr * read_imageui(image_cur, sampler, px_curtr).x +
            w_cur_bl * read_imageui(image_cur, sampler, px_curbl).x + w_cur_br * read_imageui(image_cur, sampler, px_curbr).x;
// used to compute scale for robust cost
e += fabs(res);
// robustification
float weight = res/scale_; //1.48f * vk::getMedian(errors)
chi_ += pow(res,2) * weight;
float3 j_row0 = (float3)((dx * frame_jac[0]), (dx * frame_jac[1]), (dx * frame_jac[2]));
float3 j_row1 = (float3)((dy * frame_jac[3]), (dy * frame_jac[4]), (dy * frame_jac[5]));
float3 J = (j_row0 + j_row1) * ((float)F_X / scale);
compute_hessain(J, &Hessian[f*9], weight);
Jacobian[f] -= J * res * weight;

}
}
errors[f] = e / pow(PATCH_SIZE,2.0);
chi[f]=chi_;
}
