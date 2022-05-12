/*
 *  This script is used to calculate Symmetry function features for BPNN
 *  
 *  System          : Bulk NaCl shell model
 *  Symmetry_func   : DIY symmetry func according shell model potential
 *
 *                          
 *  Date            : 2022 - 04 - 16
 *  Author          : ZaraYang
 *
 * */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

// particle number
#define nNa     540
#define nCl     540
#define nAtom   (nNa+nCl)
// Calculate parameters
#define rCut    10
#define Dim     3
#define pi      3.141592653589793
#define sqr_pi  1.772453850905516
#define nan23   6.02214076
#define ep0p12  8.8541878128
#define ele19   1.602176634
#define r4pie   0.25e6/(pi*ep0p12)*nan23*pow(ele19,2)
#define alpha   1/4.5
#define k_unit  ele19 * nan23 * 1e3                         // eV to 10 J / mol /Ans
// Particle parameters
#define ChargeNaShell   -0.50560      // unit : e
#define ChargeClShell   -2.50050        
#define ChargeNa        +1
#define ChargeCl        -1
// Matrix define
typedef vector < double > VEC;
typedef vector < vector < double >> MAT2;
typedef vector < vector < vector < double >>> MAT3;
typedef vector < vector < vector < vector < double >>>> MAT4;

// operator define
ostream& operator<<(ostream& out, const MAT2& v) {
    // print 2-D matrix to screen  
    for (int x = 0; x < v.size(); x++) {
		for (int y = 0; y < v[0].size(); y++) {
			out << v[x][y] << " ";
		}
		cout << endl;
	}
	return(out);
}
VEC operator+(const VEC& v1, const VEC& v2) {
    VEC result(3);
	for (int index = 0; index < v1.size(); index++) {
		result[index] = v1[index] + v2[index];
	}
	return(result);
}
VEC operator-(const VEC& v1, const VEC& v2) {
	VEC result(3);
	for (int index = 0; index < v1.size(); index++) {
		result[index] = v1[index] - v2[index];
	}
	return(result);
}
MAT2 operator-(const MAT2& n, const MAT2& m) {
	MAT2 result(n.size(), VEC(n[0].size()));
	for (int x = 0; x < n.size(); x++) {
		for (int y = 0; y < n[0].size(); y++) {
			result[x][y] = n[x][y] - m[x][y];
		}
	}
	return(result);
}
VEC operator*(const double c, const VEC& v) {
	VEC result(3);
	for (int index = 0; index < v.size(); index++) {
		result[index] = c * v[index];
	}
	return(result);
}
double Norm(VEC v){
	double result = 0;
	for (int index = 0; index < v.size(); index ++){
		result += v[index] * v[index];
	}
	result = sqrt(result);
	return(result);
}
// Symmetry function defination
double u_harmonic_spring(double rii, double param_k){
    /*
    *   harmonic spring potential between core and self-shell
    */
    return( 0.5 * param_k * rii * rii );
}
double du_harmonic_spring(double rii, double param_k){
    /*
    * Derivative of harmonic spring potential
    */
    return(param_k * rii);
}
double u_erfc(double rij, double erfc_alpha, double c0, double c3, double c4){
    /*
    *   short range potential of Coulumb potential a.k.a  erfc(\alpha*r) / r
    */
    double result = erfc( erfc_alpha * rij ) / rij + c0 + c3 * pow(rij, 3) + c4 * pow(rij, 4);
    return(result);
}
double du_erfc(double rij, double erfc_alpha, double c0, double c3, double c4){
    /*
    *   Derivative of short range erfc()/r
    */
    double result = 2 * erfc_alpha * exp( - erfc_alpha * erfc_alpha * rij * rij) / ( rij * sqr_pi );
    result += erfc( erfc_alpha * rij ) / (rij * rij);
    result *= -1;
    result += 3 * c3 * pow(rij, 2) + 4 * c4 * pow(rij, 3);
    return(result);
}
double u_shell(double rij, double A, double C, double D, double rho, double c0, double c3, double c4){
    double result = 0;
    result += A * exp( - rij / rho ) - ( C / pow(rij, 6) ) - ( D / pow(rij, 8));
    result += c0 + c3 * pow(rij, 3) + c4 * pow(rij, 4);
    return(result);
}
double du_shell(double rij, double A, double C, double D, double rho, double c0, double c3, double c4){
    double result = 0;
    result = - ( A / rho ) * exp( - rij / rho ) + 6 * ( C / pow(rij, 7)) + 8 * ( D / pow(rij, 9) );
    result += 3 * c3 * pow(rij, 2) + 4 * c4 * pow(rij, 3);
    return(result);
}
// Loading system coord file in VMD patten
void Load_config_data(string config_path, MAT2& Position){
	ifstream load_file(config_path,ios::in);
	string line_buffer;
	int value_index = 0;
	VEC data_buffer;

	for(int index = 0; index <6; index++){
		load_file >> line_buffer;
	}

	while(load_file >> line_buffer){
		if(value_index % 4 == 0){
			value_index += 1;
			continue;
		}
		double single_data = atof(line_buffer.c_str());
		data_buffer.push_back(single_data);
		value_index += 1;
	}
	
	for (int index = 0; index < 2 * nAtom; index ++){
		for(int dim = 0; dim < 3; dim++){
			Position[index][dim] = data_buffer[index * 3 + dim];
		    // cout << setw(16) << Position[index][dim] << "  " ;
        }
        // cout << endl;
	}
}
// Loading parameters in symmetry function
void Load_parameters(string fp_name, MAT2& param,int row_num, int col_num){
    ifstream fp(fp_name);
    for (int i = 0; i < row_num; i++)
    {
        for (int j = 0; j < col_num; j++)
        {
            fp >> param[i][j];
        }
    }
}
// Periodic Boundary Condition 
VEC PeriodicBoundaryCondition(VEC& coord, VEC& Box){
	VEC result = {0., 0., 0.};
	result[0] = coord[0] - round(coord[0]/Box[0])*Box[0];
	result[1] = coord[1] - round(coord[1]/Box[1])*Box[1];
	result[2] = coord[2] - round(coord[2]/Box[2])*Box[2];
	return(result);
}
// Calculate distance matrix
void Calculate_distance(MAT2& Position, MAT2& norm, vector<MAT2>& vec, VEC& box_range){
    VEC vec_rij = {0,0,0};
    for (int index_x = 0 ; index_x < 2 * nAtom ; index_x ++){
        for (int index_y = 0 ; index_y < 2 * nAtom ; index_y ++){
            vec_rij = Position[index_x] - Position[index_y];
            vec_rij = PeriodicBoundaryCondition(vec_rij, box_range);         
            vec[index_x][index_y] = vec_rij;
            norm[index_x][index_y] = Norm(vec_rij);
        }
    }
}
void Calculate_harmonic_spring_features(MAT2& features, MAT2& R, MAT2& params, int atom_type, string save_path){
    int range_i[] = {0, nNa};
    int range_j[] = {nAtom, nAtom + nNa};
    if(atom_type == 1){
        range_i[0] = nNa;
        range_i[1] = nAtom;
        range_j[0] = nAtom + nNa;
        range_j[1] = nAtom + nAtom;
    }
    if(atom_type == 2){
        range_i[0] = nAtom;
        range_i[1] = nAtom + nNa;
        range_j[0] = 0;
        range_j[1] = nNa;
    }
    if(atom_type == 3){
        range_i[0] = nAtom + nNa;
        range_i[1] = nAtom + nAtom;
        range_j[0] = nNa;
        range_j[1] = nAtom;
    }
    int index_i, index_j;
    for (int index = 0; index < nNa; index++){
        for (int k = 0; k < params.size();k++){
            index_i = range_i[0] + index;
            index_j = range_j[0] + index;
            features[k][index] = u_harmonic_spring(R[index_i][index_j], params[k][0] * k_unit);
        }
    }
    ofstream fp(save_path);
    for (int k = 0; k < params.size(); k++)
    {
        for (int index = 0; index < nNa; index++)
        {
            fp << features[k][index] << "\t";
        }
        fp << "\n";
    }
    fp.close();
}
void Calculate_harmonic_spring_dfeatures(MAT4& dfeatures, MAT2& R, MAT3& Vec, MAT2& params, int atom_type, string save_path){
    int range_i[] = {0, nNa};
    int range_j[] = {nAtom, nAtom + nNa};
    if(atom_type == 1){
        range_i[0] = nNa;
        range_i[1] = nAtom;
        range_j[0] = nAtom + nNa;
        range_j[1] = nAtom + nAtom;
    }
    if(atom_type == 2){
        range_i[0] = nAtom;
        range_i[1] = nAtom + nNa;
        range_j[0] = 0;
        range_j[1] = nNa;
    }
    if(atom_type == 3){
        range_i[0] = nAtom + nNa;
        range_i[1] = nAtom + nAtom;
        range_j[0] = nNa;
        range_j[1] = nAtom;
    }
    int index_i, index_j;
    for (int index = 0; index < nNa; index++){
        index_i = range_i[0] + index;
        index_j = range_j[0] + index;
        for (int k = 0; k < params.size();k++){
            dfeatures[k][index][index_i] = + du_harmonic_spring(R[index_i][index_j], params[k][0] * k_unit) * (1 / R[index_i][index_j]) * Vec[index_i][index_j];
            dfeatures[k][index][index_j] = - du_harmonic_spring(R[index_i][index_j], params[k][0] * k_unit) * (1 / R[index_i][index_j]) * Vec[index_i][index_j];
        }    
    }
    ofstream fp(save_path);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = 0; i < nNa; i++)
        {
            for (int j = 0; j < 2 * nAtom ; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << dfeatures[ip][i][j][ix] << " ";
                }
            }
        }
    }
    fp << endl;
    fp.close();
}
void Calculate_erfc_features(MAT2& features, MAT2& R, MAT2& params, int atom_type_1, int atom_type_2, string save_path){
    int range_1[] = {0, nNa};
    double charge_1 = ChargeNa - ChargeNaShell;
    if(atom_type_1 == 1){
        charge_1 = ChargeCl - ChargeClShell;
        range_1[0] = nNa;
        range_1[1] = nAtom;
    }
    if(atom_type_1 == 2){
        charge_1 = ChargeNaShell;
        range_1[0] = nAtom;
        range_1[1] = nAtom + nNa;
    }
    if(atom_type_1 == 3){
        charge_1 = ChargeClShell;
        range_1[0] = nAtom + nNa;
        range_1[1] = 2 * nAtom;
    }
    int range_2[] = {0, nNa};
    double charge_2 = ChargeNa - ChargeNaShell;
    if(atom_type_2 == 1){
        charge_2 = ChargeCl - ChargeClShell;
        range_2[0] = nNa;
        range_2[1] = nAtom;
    }
    if(atom_type_2 == 2){
        charge_2 = ChargeNaShell;
        range_2[0] = nAtom;
        range_2[1] = nAtom + nNa;
    }
    if(atom_type_2 == 3){
        charge_2 = ChargeClShell;
        range_2[0] = nAtom + nNa;
        range_2[1] = 2 * nAtom;
    }
    for (int index_1 = range_1[0]; index_1 < range_1[1]; index_1++){
        for (int index_2 = range_2[0]; index_2 < range_2[1]; index_2++){
            if(R[index_1][index_2] > rCut){continue;}
            if(index_1 == index_2 || abs(index_1 - index_2) == nAtom){continue;}
            for (int k = 0; k < params.size();k++){
                features[k][index_1 - range_1[0]] += r4pie * charge_1 * charge_2 * u_erfc(R[index_1][index_2], params[k][0], params[k][1], params[k][2], params[k][3]);
            }
        }
    }
    ofstream fp(save_path);
    for (int k = 0; k < params.size(); k++)
    {
        for (int index = range_1[0]; index < range_1[1]; index++)
        {
            fp << features[k][index - range_1[0]] << "\t";
        }
        fp << "\n";
    }
    fp.close();
}
void Calculate_erfc_dfeatures(MAT4& dfeatures, MAT2& R, MAT3& Vec, MAT2& params, int atom_type_1, int atom_type_2, string save_path){
    int range_1[] = {0, nNa};
    double charge_1 = ChargeNa - ChargeNaShell;
    if(atom_type_1 == 1){
        charge_1 = ChargeCl - ChargeClShell;
        range_1[0] = nNa;
        range_1[1] = nAtom;
    }
    if(atom_type_1 == 2){
        charge_1 = ChargeNaShell;
        range_1[0] = nAtom;
        range_1[1] = nAtom + nNa;
    }
    if(atom_type_1 == 3){
        charge_1 = ChargeClShell;
        range_1[0] = nAtom + nNa;
        range_1[1] = nAtom + nAtom;
    }
    int range_2[] = {0, nNa};
    double charge_2 = ChargeNa - ChargeNaShell;
    if(atom_type_2 == 1){
        charge_2 = ChargeCl - ChargeClShell;
        range_2[0] = nNa;
        range_2[1] = nAtom;
    }
    if(atom_type_2 == 2){
        charge_2 = ChargeNaShell;
        range_2[0] = nAtom;
        range_2[1] = nAtom + nNa;
    }
    if(atom_type_2 == 3){
        charge_2 = ChargeClShell;
        range_2[0] = nAtom + nNa;
        range_2[1] = nAtom + nAtom;
    }
    double repeat_factor = 1;
    if (atom_type_1 == atom_type_2){
        repeat_factor = 0.5;
    }
    double qij = charge_1 * charge_2;
    for (int i = range_1[0]; i < range_1[1]; i++){
        for (int j = range_2[0]; j < range_2[1]; j++){
            if(R[i][j] > rCut){continue;}
            if(i == j || abs(i - j) == nAtom){continue;}
            for (int k = 0; k < params.size();k++){
                dfeatures[k][i - range_1[0]][i] = dfeatures[k][i - range_1[0]][i] + repeat_factor * r4pie * qij * du_erfc(R[i][j], params[k][0], params[k][1], params[k][2], params[k][3]) * (1 / R[i][j]) * Vec[i][j];
                dfeatures[k][i - range_1[0]][j] = dfeatures[k][i - range_1[0]][j] - repeat_factor * r4pie * qij * du_erfc(R[i][j], params[k][0], params[k][1], params[k][2], params[k][3]) * (1 / R[i][j]) * Vec[i][j];
            }
        }
    }
    ofstream fp(save_path);
    for (int ip = 0; ip < params.size(); ip++){
        for (int i = range_1[0]; i < range_1[1]; i++)
        {
            for (int j = 0; j < 2 * nAtom ; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << dfeatures[ip][i - range_1[0]][j][ix] << " ";
                }
            }
        }
    }
    fp << endl;
    fp.close();
}
void Calculate_shell_features(MAT2& features, MAT2& R, MAT2& params, int atom_type_1, int atom_type_2, string save_path){
    double range_i[] = {nAtom, nAtom + nNa};
    if (atom_type_1 == 3){
        range_i[0] = nAtom + nNa;
        range_i[1] = nAtom + nAtom;
    }
    double range_j[] = {nAtom, nAtom + nNa};
    if (atom_type_2 == 3){
        range_j[0] = nAtom + nNa;
        range_j[1] = nAtom + nAtom;
    }
    for (int index_i = range_i[0]; index_i < range_i[1]; index_i++){
        for (int index_j = range_j[0]; index_j < range_j[1]; index_j++){
            if(R[index_i][index_j] > rCut){continue;}
            if(index_i == index_j){continue;}
            for (int k = 0; k < params.size();k++){
                features[k][index_i - range_i[0]] += u_shell(R[index_i][index_j], params[k][0] * k_unit, params[k][1] * k_unit, params[k][2] * k_unit, params[k][3], params[k][4], params[k][5], params[k][6]);
            }
        }
    }
    ofstream fp(save_path);
    for (int k = 0; k < params.size(); k++)
    {
        for (int index = range_i[0]; index < range_i[1]; index++)
        {
            fp << features[k][index - range_i[0]] << "\t";
        }
        fp << "\n";
    }
    fp.close();
}
void Calculate_shell_dfeatures(MAT4& dfeatures, MAT2& R, MAT3& Vec, MAT2& params, int atom_type_1, int atom_type_2, string save_path){
    double range_i[] = {nAtom, nAtom + nNa};
    if (atom_type_1 == 3){
        range_i[0] = nAtom + nNa;
        range_i[1] = nAtom + nAtom;
    }
    double range_j[] = {nAtom, nAtom + nNa};
    if (atom_type_2 == 3){
        range_j[0] = nAtom + nNa;
        range_j[1] = nAtom + nAtom;
    }
    double repeat_factor = 1;
    if (atom_type_1 == atom_type_2){
        repeat_factor = 0.5;
    }
    for (int i = range_i[0]; i < range_i[1]; i++){
        for (int j = range_j[0]; j < range_j[1]; j++){
            if(R[i][j] > rCut){continue;}
            if(i == j){continue;}
            for (int k = 0; k < params.size();k++){
                dfeatures[k][i - range_i[0]][i] = dfeatures[k][i - range_i[0]][i] + repeat_factor * du_shell(R[i][j], params[k][0] * k_unit, params[k][1] * k_unit, params[k][2] * k_unit, params[k][3], params[k][4], params[k][5], params[k][6]) * (1 / R[i][j]) * Vec[i][j];
                dfeatures[k][i - range_i[0]][j] = dfeatures[k][i - range_i[0]][j] - repeat_factor * du_shell(R[i][j], params[k][0] * k_unit, params[k][1] * k_unit, params[k][2] * k_unit, params[k][3], params[k][4], params[k][5], params[k][6]) * (1 / R[i][j]) * Vec[i][j];
            }
        }
    }
    ofstream fp(save_path);
    for (int ip = 0; ip < params.size(); ip++){
        for (int i = range_i[0]; i < range_i[1]; i++)
        {
            for (int j = 0; j < 2 * nAtom ; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << dfeatures[ip][i - range_i[0]][j][ix] << " ";
                }
            }
        }
    }
    fp << endl;
    fp.close();
}
void Save_2dmatrix(MAT2& mat, string save_path){
    ofstream fp(save_path);
    for (int i = 0; i < mat.size(); i++){
        for (int d = 0; d < mat[0].size(); d++){
            fp << setprecision(16) << setw(16) << mat[i][d] << "  ";
        }
        fp << endl;
    }
}
// main code
int main(){
    cout << setprecision (16);
    //  Define matrix
        MAT2 Position(2 * nAtom, VEC(3));
        MAT2 R_matrix(2 * nAtom, VEC(2 * nAtom));
        MAT3 V_matrix(2 * nAtom, MAT2(2 * nAtom, VEC(3)));
        MAT2 Force(nAtom, VEC(3));
        
        VEC Box = {24.0959889334770030,24.0959889334770030, 5 * 24.0959889334770030};
        MAT2 HS_Na_params(1, VEC(1));
        MAT2 HS_Cl_params(1, VEC(1));
        MAT2 EC_Na_params(1, VEC(4));
        MAT2 EC_Cl_params(1, VEC(4));
        
        MAT2 SH_NN_params(1, VEC(7));
        MAT2 SH_NC_params(1, VEC(7));
        MAT2 SH_CN_params(1, VEC(7));
        MAT2 SH_CC_params(1, VEC(7));

        MAT2 features_HS_Nn(HS_Na_params.size(), VEC(nNa));
        MAT2 features_HS_Cc(HS_Cl_params.size(), VEC(nCl));
        
        MAT2 features_EC_NN(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_NC(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_CN(EC_Na_params.size(), VEC(nCl));
        MAT2 features_EC_CC(EC_Cl_params.size(), VEC(nCl));
        
        MAT2 features_EC_Nn(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_Nc(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_Cn(EC_Cl_params.size(), VEC(nCl));
        MAT2 features_EC_Cc(EC_Cl_params.size(), VEC(nCl));
        
        MAT2 features_EC_nn(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_nc(EC_Na_params.size(), VEC(nNa));
        MAT2 features_EC_cn(EC_Na_params.size(), VEC(nCl));
        MAT2 features_EC_cc(EC_Cl_params.size(), VEC(nCl));

        MAT2 features_SH_nn(SH_NN_params.size(), VEC(nNa));
        MAT2 features_SH_nc(SH_NC_params.size(), VEC(nNa));
        MAT2 features_SH_cn(SH_NC_params.size(), VEC(nCl));
        MAT2 features_SH_cc(SH_CC_params.size(), VEC(nCl));

        MAT4 dfeatures_HS_Nn(HS_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom,VEC(3))));
        MAT4 dfeatures_HS_Cc(HS_Cl_params.size(), MAT3(nCl, MAT2(2 * nAtom,VEC(3))));
       
        MAT4 dfeatures_EC_NN(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_NC(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_CN(EC_Na_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_CC(EC_Cl_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        
        MAT4 dfeatures_EC_Nn(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_Nc(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_Cn(EC_Cl_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_Cc(EC_Cl_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        
        MAT4 dfeatures_EC_nn(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_nc(EC_Na_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_cn(EC_Na_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_EC_cc(EC_Cl_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));

        MAT4 dfeatures_SH_nn(SH_NN_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_SH_nc(SH_NC_params.size(), MAT3(nNa, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_SH_cn(SH_NC_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));
        MAT4 dfeatures_SH_cc(SH_CC_params.size(), MAT3(nCl, MAT2(2 * nAtom, VEC(3))));

        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/HS_Cl_params.txt",HS_Cl_params,1,1);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/HS_Na_params.txt",HS_Na_params,1,1);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/EC_Na_params.txt",EC_Na_params,1,4);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/EC_Cl_params.txt",EC_Cl_params,1,4);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/SH_NN_params.txt",SH_NN_params,1,7);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/SH_NC_params.txt",SH_NC_params,1,7);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/SH_NC_params.txt",SH_CN_params,1,7);
        Load_parameters("/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/NaCl_BPNN/code/Params/SH_CC_params.txt",SH_CC_params,1,7);
    
    cout << EC_Na_params << endl;    
    //  Loading Particles position
        Load_config_data("./Config.xyz",Position); 
    // Calculate distance between particles
        Calculate_distance(Position, R_matrix, V_matrix, Box);
    // Calculate harmonic spring features
        Calculate_harmonic_spring_features(features_HS_Nn, R_matrix, HS_Na_params, 0, "./features/feature_HS_Nn.txt");
        Calculate_harmonic_spring_features(features_HS_Cc, R_matrix, HS_Cl_params, 1, "./features/feature_HS_Cc.txt");
    // Calculate Coulumb erfc features
        Calculate_erfc_features(features_EC_NN, R_matrix, EC_Na_params, 0, 0, "./features/feature_EC_NN.txt");
        Calculate_erfc_features(features_EC_NC, R_matrix, EC_Na_params, 0, 1, "./features/feature_EC_NC.txt");  
        Calculate_erfc_features(features_EC_CN, R_matrix, EC_Cl_params, 1, 0, "./features/feature_EC_CN.txt");  
        Calculate_erfc_features(features_EC_CC, R_matrix, EC_Cl_params, 1, 1, "./features/feature_EC_CC.txt"); 
        
        Calculate_erfc_features(features_EC_Nn, R_matrix, EC_Na_params, 0, 2, "./features/feature_EC_Nn.txt");
        Calculate_erfc_features(features_EC_Nc, R_matrix, EC_Na_params, 0, 3, "./features/feature_EC_Nc.txt");
        Calculate_erfc_features(features_EC_Cn, R_matrix, EC_Cl_params, 1, 2, "./features/feature_EC_Cn.txt");
        Calculate_erfc_features(features_EC_Cc, R_matrix, EC_Cl_params, 1, 3, "./features/feature_EC_Cc.txt");
       
        Calculate_erfc_features(features_EC_nn, R_matrix, EC_Na_params, 2, 2, "./features/feature_EC_nn.txt");
        Calculate_erfc_features(features_EC_nc, R_matrix, EC_Na_params, 2, 3, "./features/feature_EC_nc.txt");
        Calculate_erfc_features(features_EC_cn, R_matrix, EC_Na_params, 3, 2, "./features/feature_EC_cn.txt");
        Calculate_erfc_features(features_EC_cc, R_matrix, EC_Cl_params, 3, 3, "./features/feature_EC_cc.txt");     
    // Calculate shell features
        Calculate_shell_features(features_SH_nn, R_matrix, SH_NN_params, 2, 2, "./features/feature_SH_nn.txt");
        Calculate_shell_features(features_SH_nc, R_matrix, SH_NC_params, 2, 3, "./features/feature_SH_nc.txt");
        Calculate_shell_features(features_SH_cn, R_matrix, SH_NC_params, 3, 2, "./features/feature_SH_cn.txt");
        Calculate_shell_features(features_SH_cc, R_matrix, SH_CC_params, 3, 3, "./features/feature_SH_cc.txt");
    // Calculate harmonic spring dfeatures
        Calculate_harmonic_spring_dfeatures(dfeatures_HS_Nn, R_matrix, V_matrix, HS_Na_params, 0, "./features/dfeature_HS_Nn.txt"); 
        Calculate_harmonic_spring_dfeatures(dfeatures_HS_Cc, R_matrix, V_matrix, HS_Cl_params, 1, "./features/dfeature_HS_Cc.txt");
    // Calculate Coulumb erfc dfeatures
        Calculate_erfc_dfeatures(dfeatures_EC_NN, R_matrix, V_matrix, EC_Na_params, 0, 0, "./features/dfeature_EC_NN.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_NC, R_matrix, V_matrix, EC_Na_params, 0, 1, "./features/dfeature_EC_NC.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_CN, R_matrix, V_matrix, EC_Na_params, 1, 0, "./features/dfeature_EC_CN.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_CC, R_matrix, V_matrix, EC_Cl_params, 1, 1, "./features/dfeature_EC_CC.txt");
        
        Calculate_erfc_dfeatures(dfeatures_EC_Nn, R_matrix, V_matrix, EC_Na_params, 0, 2, "./features/dfeature_EC_Nn.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_Nc, R_matrix, V_matrix, EC_Na_params, 0, 3, "./features/dfeature_EC_Nc.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_Cn, R_matrix, V_matrix, EC_Cl_params, 1, 2, "./features/dfeature_EC_Cn.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_Cc, R_matrix, V_matrix, EC_Cl_params, 1, 3, "./features/dfeature_EC_Cc.txt");
       
        Calculate_erfc_dfeatures(dfeatures_EC_nn, R_matrix, V_matrix, EC_Na_params, 2, 2, "./features/dfeature_EC_nn.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_nc, R_matrix, V_matrix, EC_Na_params, 2, 3, "./features/dfeature_EC_nc.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_cn, R_matrix, V_matrix, EC_Cl_params, 3, 2, "./features/dfeature_EC_cn.txt");
        Calculate_erfc_dfeatures(dfeatures_EC_cc, R_matrix, V_matrix, EC_Cl_params, 3, 3, "./features/dfeature_EC_cc.txt");
    // Calculate shell dfeatures
        Calculate_shell_dfeatures(dfeatures_SH_nn, R_matrix, V_matrix, SH_NN_params, 2, 2, "./features/dfeature_SH_nn.txt");
        Calculate_shell_dfeatures(dfeatures_SH_nc, R_matrix, V_matrix, SH_NC_params, 2, 3, "./features/dfeature_SH_nc.txt");
        Calculate_shell_dfeatures(dfeatures_SH_cn, R_matrix, V_matrix, SH_NC_params, 3, 2, "./features/dfeature_SH_cn.txt");
        Calculate_shell_dfeatures(dfeatures_SH_cc, R_matrix, V_matrix, SH_CC_params, 3, 3, "./features/dfeature_SH_cc.txt");
    // Calculate force 
        MAT2 TForce( 2 * nAtom, VEC(3));
        for(int i = 0; i < 2 * nAtom; i++){
            TForce[i] = {0., 0., 0.};
         }
        for (int i = 0; i < nNa; i++){
            for ( int j = 0; j < 2 * nAtom; j++){
                for (int d = 0; d < 3; d++){
                    TForce[j][d] -= dfeatures_HS_Nn[0][i][j][d];
                    TForce[j][d] -= dfeatures_HS_Cc[0][i][j][d];
                    
                    TForce[j][d] -= dfeatures_EC_NN[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_CN[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_CC[0][i][j][d];

                    TForce[j][d] -= dfeatures_EC_Nn[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_Nc[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_Cn[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_Cc[0][i][j][d];
                    
                    TForce[j][d] -= dfeatures_EC_nn[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_cn[0][i][j][d];
                    TForce[j][d] -= dfeatures_EC_cc[0][i][j][d];
                
                    TForce[j][d] -= dfeatures_SH_nn[0][i][j][d];
                    TForce[j][d] -= dfeatures_SH_cn[0][i][j][d];
                    TForce[j][d] -= dfeatures_SH_cc[0][i][j][d];
                }
            }
        }
         
        ofstream fp_N("./Nforce_cal.txt");
        ofstream fp_C("./Cforce_cal.txt");
        ofstream fp_n("./nforce_cal.txt");
        ofstream fp_c("./cforce_cal.txt");
             
        for (int i = 0; i < nNa; i++){
            for (int d = 0; d < 3; d++){ 
                fp_N << setprecision(16) << setw(16) << TForce[i][d] << "  ";
                fp_C << setprecision(16) << setw(16) << TForce[i + nNa][d] << "  ";
                fp_n << setprecision(16) << setw(16) << TForce[i + nAtom][d] << "  ";
                fp_c << setprecision(16) << setw(16) << TForce[i + nAtom + nNa][d] << "  ";
            }
            fp_N << endl;
            fp_C << endl;
            fp_n << endl;
            fp_c << endl;
        }
}






