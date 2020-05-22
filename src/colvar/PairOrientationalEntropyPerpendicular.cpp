/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Colvar.h"
#include "ActionRegister.h"
#include "tools/NeighborListParallel.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
#include "tools/IFile.h"

#include <string>
#include <math.h>

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR PAIRENTROPY
/*
Calculate the global pair entropy using the expression:
\f[
s=-2\pi\rho k_B \int\limits_0^{r_{\mathrm{max}}} \left [ g(r) \ln g(r) - g(r) + 1 \right ] r^2 dr .
\f]
where \f$ g(r) $\f is the pair distribution function and \f$ r_{\mathrm{max}} $\f is a cutoff in the integration (MAXR).
For the integration the interval from 0 to  \f$ r_{\mathrm{max}} $\f is partitioned in NHIST equal intervals. 
To make the calculation of \f$ g(r) $\f differentiable, the following function is used:
\f[
g(r) = \frac{1}{4 \pi \rho r^2} \sum\limits_{j} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(r-r_{ij})^2/(2\sigma^2)} ,
\f]
where \f$ \rho $\f is the density and \f$ sigma $\f is a broadening parameter (SIGMA).  
\par Example)
The following input tells plumed to calculate the pair entropy of atoms 1-250 with themselves.
\verbatim
PAIRENTROPY ...
 LABEL=s2
 GROUPA=1-250
 MAXR=0.65
 SIGMA=0.025
 NHIST=100
 NLIST
 NL_CUTOFF=0.75
 NL_STRIDE=10
... PAIRENTROPY
\endverbatim
*/
//+ENDPLUMEDOC

class PairOrientationalEntropyPerpendicular : public Colvar {
  bool pbc, serial, invalidateList, firsttime, doneigh;
  NeighborListParallel *nl;
  vector<AtomNumber> center_lista,start1_lista,end1_lista,start2_lista,end2_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  double maxr;
  vector<int> nhist_;
  int nhist1_nhist2_;
  vector<double> sigma_;
  double rcut2;
  double invTwoPiSigma1Sigma2, sigma1Sqr, sigma2Sqr, twoSigma1Sqr,twoSigma2Sqr;
  double deltar, deltaAngle, deltaCosAngle;
  unsigned deltaBin, deltaBinAngle;
  // Integration routines
  double integrate(Matrix<double> integrand, vector<double> delta)const;
  Vector integrate(Matrix<Vector> integrand, vector<double> delta)const;
  Tensor integrate(Matrix<Tensor> integrand, vector<double> delta)const;
  vector<double> x1, x2, x1sqr, x2sqr;
  // Kernel to calculate g(r)
  double kernel(vector<double> distance, double invNormKernel, vector<double>&der)const;
  // Output gofr and integrand
  void outputGofr(Matrix<double> gofr, const char* fileName);
  void outputIntegrand(vector<double> integrand);
  int outputStride;
  bool doOutputGofr, doOutputIntegrand;
  mutable PLMD::OFile gofrOfile;
  // Reference g(r)
  bool doReferenceGofr;
  Matrix<double> referenceGofr;
  double epsilon;
  double densityReference;
  // Average gofr
  Matrix<double> avgGofr;
  unsigned iteration;
  bool doAverageGofr;
  unsigned averageGofrTau;
  // Up-down symmetry
  bool doUpDownSymmetry;
  double startCosAngle;
  // Low communication variant
  bool doLowComm;
public:
  explicit PairOrientationalEntropyPerpendicular(const ActionOptions&);
  ~PairOrientationalEntropyPerpendicular();
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(PairOrientationalEntropyPerpendicular,"PAIR_ORIENTATIONAL_ENTROPY_PERPENDICULAR")

void PairOrientationalEntropyPerpendicular::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r)");
  keys.addFlag("AVERAGE_GOFR",false,"Average g(r) over time");
  keys.add("optional","AVERAGE_GOFR_TAU","Characteristic length of a window in which to average the g(r). It is in units of iterations and should be an integer. Zero corresponds to an normal average (infinite window).");
  keys.addFlag("UP_DOWN_SYMMETRY",false,"The symmetry is such that parallel and antiparallel vectors are not distinguished. The angle goes from 0 to pi/2 instead of from 0 to pi.");
  keys.add("optional","OUTPUT_STRIDE","The frequency with which the output is written to files");
  keys.addFlag("OUTPUT_INTEGRAND",false,"Output integrand");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","ORIGIN","Define an atom that represents the origin from which to calculate the g(r,theta)");
  keys.add("atoms","CENTER","Center atoms");
  keys.add("atoms","START1","Start point of first vector defining orientation");
  keys.add("atoms","START2","Start point of second vector defining orientation");
  keys.add("atoms","END1","End point of first vector defining orientation");
  keys.add("atoms","END2","End point of second vector defining orientation");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("optional","NHIST","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
  keys.add("optional","REFERENCE_GOFR_FNAME","the name of the file with the reference g(r)");
  keys.add("optional","REFERENCE_DENSITY","Density to be used with the reference g(r). If not specified or less than 0, the current density is used. Using the current density might lead in large changes of the box volume.");
  keys.addFlag("LOW_COMM",false,"Use an algorithm with less communication between processors");
}

PairOrientationalEntropyPerpendicular::PairOrientationalEntropyPerpendicular(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("CENTER",center_lista);
  parseAtomList("START1",start1_lista);
  parseAtomList("END1",end1_lista);
  parseAtomList("START2",start2_lista);
  parseAtomList("END2",end2_lista);
  if(center_lista.size()!=start1_lista.size()) error("Number of atoms in START1 must be equal to the number of atoms in CENTER");
  if(center_lista.size()!=end1_lista.size()) error("Number of atoms in END1 must be equal to the number of atoms in CENTER");
  if(center_lista.size()!=start2_lista.size()) error("Number of atoms in START2 must be equal to the number of atoms in CENTER");
  if(center_lista.size()!=end2_lista.size()) error("Number of atoms in END2 must be equal to the number of atoms in CENTER");

  bool nopbc=!pbc;
  pbc=!nopbc;

// neighbor list stuff
  doneigh=false;
  bool nl_full_list=false;
  double nl_cut=0.0;
  double nl_skin;
  int nl_st=-1;
  parseFlag("NLIST",doneigh);
  if(doneigh){
   parse("NL_CUTOFF",nl_cut);
   if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
   parse("NL_STRIDE",nl_st);
   //if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  addValueWithDerivatives(); setNotPeriodic();

  parse("MAXR",maxr);
  log.printf("  Integration in the interval from 0. to %f \n", maxr );

  parseVector("SIGMA",sigma_);
  if(sigma_.size() != 2) error("SIGMA keyword takes two input values");
  log.printf("  The pair distribution function is calculated with a Gaussian kernel with deviations %f and %f \n", sigma_[0], sigma_[1]);
  double rcut = maxr + 2*sigma_[0];  // 2*sigma is hard coded
  rcut2 = rcut*rcut;
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 2*SIGMA");
    nl_skin=nl_cut-rcut;
  }

  doUpDownSymmetry=false;
  parseFlag("UP_DOWN_SYMMETRY",doUpDownSymmetry);
  if (doUpDownSymmetry) log.printf("  The angle can take values between 0 and pi/2 due to the up down symmetry. \n");

  parseVector("NHIST",nhist_);
  if (nhist_.size()<1) {
     nhist_.resize(2);
     // Default values
     nhist_[0]=ceil(maxr/sigma_[0]) + 1; 
     if (doUpDownSymmetry) nhist_[1]=ceil(1./sigma_[1]) + 1;
     else nhist_[1]=ceil(2./sigma_[1]) + 1;
  }
  if(nhist_.size() != 2) error("NHIST keyword takes two input values");
  nhist1_nhist2_=nhist_[0]*nhist_[1];
  log.printf("  The r-theta space is discretized using a grid of size %u times %u. \n", nhist_[0], nhist_[1] );
  log.printf("  The integration is performed with the trapezoid rule. \n");

  doOutputGofr=false;
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  if (doOutputGofr) { 
     log.printf("  The g(r) will be written to a file \n");
     gofrOfile.link(*this);
     gofrOfile.open("gofr.txt");
  }
  doOutputIntegrand=false;
  parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);
  if (doOutputIntegrand) {
     log.printf("  The integrand will be written to a file \n");
  }
  outputStride=1;
  parse("OUTPUT_STRIDE",outputStride);
  if (outputStride!=1 && !doOutputGofr && !doOutputIntegrand) error("Cannot specify OUTPUT_STRIDE if OUTPUT_GOFR or OUTPUT_INTEGRAND not used");
  if (outputStride<1) error("The output stride specified with OUTPUT_STRIDE must be greater than or equal to one.");
  if (outputStride>1) log.printf("  The output stride to write g(r) or the integrand is %d \n", outputStride);

  densityReference=-1.;
  parse("REFERENCE_DENSITY",densityReference);
  if (densityReference>0) log.printf("  Using a density reference of %f .\n", densityReference);

  doReferenceGofr=false;
  std::string referenceGofrFileName;
  parse("REFERENCE_GOFR_FNAME",referenceGofrFileName); 
  if (!referenceGofrFileName.empty() ) {
    epsilon=1.e-8;
    log.printf("  Reading a reference g(r) from the file %s . \n", referenceGofrFileName.c_str() );
    doReferenceGofr=true;
    IFile ifile; 
    ifile.link(*this);
    ifile.open(referenceGofrFileName);
    referenceGofr.resize(nhist_[0],nhist_[1]);
    for(unsigned int i=0;i<nhist_[0];i++) {
       for(unsigned int j=0;j<nhist_[1];j++) {
       double tmp_r, tmp_theta;
       ifile.scanField("r",tmp_r).scanField("theta",tmp_theta).scanField("gofr",referenceGofr[i][j]).scanField();
       }
    }
  }

  doAverageGofr=false;
  parseFlag("AVERAGE_GOFR",doAverageGofr);
  if (doAverageGofr) {
     iteration = 1;
     avgGofr.resize(nhist_[0],nhist_[1]);
  }
  averageGofrTau=0;
  parse("AVERAGE_GOFR_TAU",averageGofrTau);
  if (averageGofrTau!=0 && !doAverageGofr) error("AVERAGE_GOFR_TAU specified but AVERAGE_GOFR not given. Specify AVERAGE_GOFR or remove AVERAGE_GOFR_TAU");
  if (doAverageGofr && averageGofrTau==0) log.printf("The g(r) will be averaged over all frames \n");
  if (doAverageGofr && averageGofrTau!=0) log.printf("The g(r) will be averaged with a window of %d steps \n", averageGofrTau);



  doLowComm=false;
  parseFlag("LOW_COMM",doLowComm);
  if (doLowComm) {
     log.printf("  Using the low communication variant of the algorithm");
     nl_full_list=true;
  }

  checkRead();

  // Neighbor lists
  if (doneigh) {
    nl= new NeighborListParallel(center_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin);
    log.printf("  using neighbor lists with\n");
    log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
    if(nl_st>=0){
      log.printf("  update every %d steps\n",nl_st);
    } else {
      log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
    }
  }
  atomsToRequest.reserve ( center_lista.size() + start1_lista.size() + end1_lista.size() + start2_lista.size() + end2_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start1_lista.begin(), start1_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end1_lista.begin(), end1_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start2_lista.begin(), start2_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end2_lista.begin(), end2_lista.end() );
  requestAtoms(atomsToRequest);

  // Define heavily used expressions
  invTwoPiSigma1Sigma2 = (1./(2.*pi*sigma_[0]*sigma_[1]));
  sigma1Sqr = sigma_[0]*sigma_[0];
  sigma2Sqr = sigma_[1]*sigma_[1];
  twoSigma1Sqr = 2*sigma_[0]*sigma_[0];
  twoSigma2Sqr = 2*sigma_[1]*sigma_[1];
  deltar=maxr/(nhist_[0]-1);
  if (!doUpDownSymmetry) {
     deltaCosAngle=2./(nhist_[1]-1);
     startCosAngle=-1.;
  }
  else {
     deltaCosAngle=1./(nhist_[1]-1);
     startCosAngle=0.;
  }
  deltaBin = std::floor(2*sigma_[0]/deltar); // 2*sigma is hard coded
  deltaBinAngle = std::floor(2*sigma_[1]/deltaCosAngle); // 2*sigma is hard coded

  x1.resize(nhist_[0]);
  x1sqr.resize(nhist_[0]);
  x2.resize(nhist_[1]);
  x2sqr.resize(nhist_[1]);
  for(unsigned i=0;i<nhist_[0];++i){
     x1[i]=deltar*i;
     x1sqr[i]=x1[i]*x1[i];
  }
  for(unsigned i=0;i<nhist_[1];++i){
     x2[i]=startCosAngle+deltaCosAngle*i;
     x2sqr[i]=x2[i]*x2[i];
  }
}

PairOrientationalEntropyPerpendicular::~PairOrientationalEntropyPerpendicular(){
  if (doneigh) {
     nl->printStats();
     delete nl;
  }
  if (doOutputGofr) gofrOfile.close();
}

void PairOrientationalEntropyPerpendicular::prepare(){
  if(doneigh && nl->getStride()>0){
    if(firsttime) {
      invalidateList=true;
      firsttime=false;
    } else if ( (nl->getStride()>=0) &&  (getStep()%nl->getStride()==0) ){
      invalidateList=true;
    } else if ( (nl->getStride()<0) && !(nl->isListStillGood(getPositions())) ){
      invalidateList=true;
    } else {
      invalidateList=false;
    }
  }
}

// calculator
void PairOrientationalEntropyPerpendicular::calculate()
{
  //clock_t begin_time = clock();
  // Define intermediate quantities
  Matrix<double> gofr(nhist_[0],nhist_[1]);
  vector<Vector> gofrPrimeCenter(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStart1(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEnd1(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStart2(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEnd2(nhist_[0]*nhist_[1]*center_lista.size());
  Matrix<Tensor> gofrVirial(nhist_[0],nhist_[1]);
  // Calculate volume and density
  double volume=getBox().determinant();
  double density=center_lista.size()/volume;
  // Normalization of g(r)
  double normConstantBase = 2*pi*center_lista.size()*density;
  normConstantBase /= invTwoPiSigma1Sigma2;
  // Take into account "volume" of angles
  double volumeOfAngles;
  if (!doUpDownSymmetry) volumeOfAngles = 2.;
  else volumeOfAngles = 1.;
  normConstantBase /= volumeOfAngles;
  double invNormConstantBase = 1./normConstantBase;
  // Setup parallelization
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }
  if (doneigh && doLowComm) {
    if(invalidateList){
      vector<Vector> centerPositions(getPositions().begin(),getPositions().begin() + center_lista.size());
      nl->update(centerPositions);
    }
    for(unsigned int i=0;i<nl->getNumberOfLocalAtoms();i+=1) {

       unsigned index=nl->getIndexOfLocalAtom(i);
       unsigned atom1_mol1=index+center_lista.size();
       unsigned atom2_mol1=index+2*center_lista.size();
       unsigned atom3_mol1=index+3*center_lista.size();
       unsigned atom4_mol1=index+4*center_lista.size();
       std::vector<unsigned> neighbors=nl->getNeighbors(index);
 
       Vector position_index=getPosition(index);
       Vector mol1_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
       Vector mol1_vector2=pbcDistance(getPosition(atom3_mol1),getPosition(atom4_mol1)); 
       Vector mol_vector1=crossProduct(mol1_vector1,mol1_vector2); 
       double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
       double inv_v1=1./norm_v1;
       double inv_v1_sqr=inv_v1*inv_v1;

       // Loop over neighbors
       for(unsigned int j=0;j<neighbors.size();j+=1) {  
          unsigned neighbor=neighbors[j];
          if(getAbsoluteIndex(index)==getAbsoluteIndex(neighbor)) continue;
          Vector distance=pbcDistance(position_index,getPosition(neighbor));
          double d2;
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);

             unsigned atom1_mol2=neighbor+center_lista.size();
             unsigned atom2_mol2=neighbor+2*center_lista.size();
             unsigned atom3_mol2=neighbor+3*center_lista.size();
             unsigned atom4_mol2=neighbor+4*center_lista.size();
          
             Vector mol2_vector1=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2)); 
             Vector mol2_vector2=pbcDistance(getPosition(atom3_mol2),getPosition(atom4_mol2)); 
             Vector mol_vector2=crossProduct(mol2_vector1,mol2_vector2); 
             double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
             double inv_v2=1./norm_v2;
             double inv_v1_inv_v2=inv_v1*inv_v2;
             double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
             
             Vector der_cosTheta_mol1_vector1;
             der_cosTheta_mol1_vector1[0]=(-mol1_vector2[2]*mol_vector2[1]+mol1_vector2[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[2]*mol_vector1[1]+mol1_vector2[1]*mol_vector1[2]);
             der_cosTheta_mol1_vector1[1]=( mol1_vector2[2]*mol_vector2[0]-mol1_vector2[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector2[2]*mol_vector1[0]-mol1_vector2[0]*mol_vector1[2]);
             der_cosTheta_mol1_vector1[2]=(-mol1_vector2[1]*mol_vector2[0]+mol1_vector2[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[1]*mol_vector1[0]+mol1_vector2[0]*mol_vector1[1]);
             
             Vector der_cosTheta_mol1_vector2;
             der_cosTheta_mol1_vector2[0]=( mol1_vector1[2]*mol_vector2[1]-mol1_vector1[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[2]*mol_vector1[1]-mol1_vector1[1]*mol_vector1[2]);
             der_cosTheta_mol1_vector2[1]=(-mol1_vector1[2]*mol_vector2[0]+mol1_vector1[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector1[2]*mol_vector1[0]+mol1_vector1[0]*mol_vector1[2]);
             der_cosTheta_mol1_vector2[2]=( mol1_vector1[1]*mol_vector2[0]-mol1_vector1[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[1]*mol_vector1[0]-mol1_vector1[0]*mol_vector1[1]);
             
             if (doUpDownSymmetry && cosAngle<0) {
                der_cosTheta_mol1_vector1 *= -1.;
                der_cosTheta_mol1_vector2 *= -1.;
             }
             
             unsigned binAngle;
             if (doUpDownSymmetry && cosAngle<0) {
                binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
             } else {
                binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
             }
             int minBin, maxBin; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
             minBin=bin - deltaBin;
             if (minBin < 0) minBin=0;
             if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
             maxBin=bin +  deltaBin;
             if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
             int minBinAngle, maxBinAngle; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
             minBinAngle=binAngle - deltaBinAngle;
             maxBinAngle=binAngle +  deltaBinAngle;
             for(int k=minBin;k<maxBin+1;k+=1) {
               double invNormKernel=invNormConstantBase/x1sqr[k];
               vector<double> pos(2);
               pos[0]=x1[k]-distanceModulo;
               for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
                  double theta=startCosAngle+deltaCosAngle*l;
                  if (doUpDownSymmetry && cosAngle<0) {
                     pos[1]=theta+cosAngle;
                  } else {
                     pos[1]=theta-cosAngle;
                  }
                  // Include periodic effects
                  int h;
                  if (l<0) {
                     h=-l;
                  } else if (l>(nhist_[1]-1)) {
                     h=2*nhist_[1]-l-2;
                  } else {
                     h=l;
                  }
                  Vector value1;
                  vector<double> dfunc(2);
                  if (l==(nhist_[1]-1) || l==0) {
                     gofr[k][h] += kernel(pos,2*invNormKernel,dfunc)/2.;
                  } else {
                     gofr[k][h] += kernel(pos,invNormKernel,dfunc)/2.;
                  }
                  value1 = dfunc[0]*distance_versor;
                  Vector value2_mol1_vector1 = dfunc[1]*der_cosTheta_mol1_vector1;
                  Vector value2_mol1_vector2 = dfunc[1]*der_cosTheta_mol1_vector2;
                    
                  gofrPrimeCenter[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                  gofrPrimeStart1[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector1;
                  gofrPrimeEnd1[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector1;
                  gofrPrimeStart2[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector2;
                  gofrPrimeEnd2[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector2;
             
                  Tensor vv1(value1, distance);
                  Tensor vv2_mol1_vector1(value2_mol1_vector1, mol1_vector1);
                  Tensor vv2_mol1_vector2(value2_mol1_vector2, mol1_vector2);
             
                  gofrVirial[k][h] += vv1/2. + vv2_mol1_vector1 + vv2_mol1_vector2;
               }
            }
          }
        }
     }
  } else if (doneigh && !doLowComm) {
    if(invalidateList){
      vector<Vector> centerPositions(getPositions().begin(),getPositions().begin() + center_lista.size());
      nl->update(centerPositions);
    }
    for(unsigned int i=0;i<nl->getNumberOfLocalAtoms();i+=1) {

       unsigned index=nl->getIndexOfLocalAtom(i);
       unsigned atom1_mol1=index+center_lista.size();
       unsigned atom2_mol1=index+2*center_lista.size();
       unsigned atom3_mol1=index+3*center_lista.size();
       unsigned atom4_mol1=index+4*center_lista.size();
       std::vector<unsigned> neighbors=nl->getNeighbors(index);
 
       Vector position_index=getPosition(index);
       Vector mol1_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
       Vector mol1_vector2=pbcDistance(getPosition(atom3_mol1),getPosition(atom4_mol1)); 
       Vector mol_vector1=crossProduct(mol1_vector1,mol1_vector2); 
       double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
       double inv_v1=1./norm_v1;
       double inv_v1_sqr=inv_v1*inv_v1;

       // Loop over neighbors
       for(unsigned int j=0;j<neighbors.size();j+=1) {  
          unsigned neighbor=neighbors[j];
          Vector distance=pbcDistance(position_index,getPosition(neighbor));
          double d2;
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);

             unsigned atom1_mol2=neighbor+center_lista.size();
             unsigned atom2_mol2=neighbor+2*center_lista.size();
             unsigned atom3_mol2=neighbor+3*center_lista.size();
             unsigned atom4_mol2=neighbor+4*center_lista.size();
          
             Vector mol2_vector1=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2)); 
             Vector mol2_vector2=pbcDistance(getPosition(atom3_mol2),getPosition(atom4_mol2)); 
             Vector mol_vector2=crossProduct(mol2_vector1,mol2_vector2); 
             double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
             double inv_v2=1./norm_v2;
             double inv_v2_sqr=inv_v2*inv_v2;
             double inv_v1_inv_v2=inv_v1*inv_v2;
             double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
             
             Vector der_cosTheta_mol1_vector1;
             der_cosTheta_mol1_vector1[0]=(-mol1_vector2[2]*mol_vector2[1]+mol1_vector2[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[2]*mol_vector1[1]+mol1_vector2[1]*mol_vector1[2]);
             der_cosTheta_mol1_vector1[1]=( mol1_vector2[2]*mol_vector2[0]-mol1_vector2[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector2[2]*mol_vector1[0]-mol1_vector2[0]*mol_vector1[2]);
             der_cosTheta_mol1_vector1[2]=(-mol1_vector2[1]*mol_vector2[0]+mol1_vector2[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[1]*mol_vector1[0]+mol1_vector2[0]*mol_vector1[1]);
             
             Vector der_cosTheta_mol1_vector2;
             der_cosTheta_mol1_vector2[0]=( mol1_vector1[2]*mol_vector2[1]-mol1_vector1[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[2]*mol_vector1[1]-mol1_vector1[1]*mol_vector1[2]);
             der_cosTheta_mol1_vector2[1]=(-mol1_vector1[2]*mol_vector2[0]+mol1_vector1[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector1[2]*mol_vector1[0]+mol1_vector1[0]*mol_vector1[2]);
             der_cosTheta_mol1_vector2[2]=( mol1_vector1[1]*mol_vector2[0]-mol1_vector1[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[1]*mol_vector1[0]-mol1_vector1[0]*mol_vector1[1]);
             
             Vector der_cosTheta_mol2_vector1;
             der_cosTheta_mol2_vector1[0]=(-mol2_vector2[2]*mol_vector1[1]+mol2_vector2[1]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector2[2]*mol_vector2[1]+mol2_vector2[1]*mol_vector2[2]);
             der_cosTheta_mol2_vector1[1]=( mol2_vector2[2]*mol_vector1[0]-mol2_vector2[0]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector2[2]*mol_vector2[0]-mol2_vector2[0]*mol_vector2[2]);
             der_cosTheta_mol2_vector1[2]=(-mol2_vector2[1]*mol_vector1[0]+mol2_vector2[0]*mol_vector1[1])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector2[1]*mol_vector2[0]+mol2_vector2[0]*mol_vector2[1]);
             
             Vector der_cosTheta_mol2_vector2;
             der_cosTheta_mol2_vector2[0]=( mol2_vector1[2]*mol_vector1[1]-mol2_vector1[1]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector1[2]*mol_vector2[1]-mol2_vector1[1]*mol_vector2[2]);
             der_cosTheta_mol2_vector2[1]=(-mol2_vector1[2]*mol_vector1[0]+mol2_vector1[0]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector1[2]*mol_vector2[0]+mol2_vector1[0]*mol_vector2[2]);
             der_cosTheta_mol2_vector2[2]=( mol2_vector1[1]*mol_vector1[0]-mol2_vector1[0]*mol_vector1[1])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector1[1]*mol_vector2[0]-mol2_vector1[0]*mol_vector2[1]);
             
             if (doUpDownSymmetry && cosAngle<0) {
                der_cosTheta_mol1_vector1 *= -1.;
                der_cosTheta_mol1_vector2 *= -1.;
                der_cosTheta_mol2_vector1 *= -1.;
                der_cosTheta_mol2_vector2 *= -1.;
             }
             
             unsigned binAngle;
             if (doUpDownSymmetry && cosAngle<0) {
                binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
             } else {
                binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
             }
             int minBin, maxBin; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
             minBin=bin - deltaBin;
             if (minBin < 0) minBin=0;
             if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
             maxBin=bin +  deltaBin;
             if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
             int minBinAngle, maxBinAngle; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
             minBinAngle=binAngle - deltaBinAngle;
             maxBinAngle=binAngle +  deltaBinAngle;
             for(int k=minBin;k<maxBin+1;k+=1) {
               double invNormKernel=invNormConstantBase/x1sqr[k];
               vector<double> pos(2);
               pos[0]=x1[k]-distanceModulo;
               for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
                  double theta=startCosAngle+deltaCosAngle*l;
                  if (doUpDownSymmetry && cosAngle<0) {
                     pos[1]=theta+cosAngle;
                  } else {
                     pos[1]=theta-cosAngle;
                  }
                  // Include periodic effects
                  int h;
                  if (l<0) {
                     h=-l;
                  } else if (l>(nhist_[1]-1)) {
                     h=2*nhist_[1]-l-2;
                  } else {
                     h=l;
                  }
                  Vector value1;
                  vector<double> dfunc(2);
                  if (l==(nhist_[1]-1) || l==0) {
                     gofr[k][h] += kernel(pos,2*invNormKernel,dfunc);
                  } else {
                     gofr[k][h] += kernel(pos,invNormKernel,dfunc);
                  }
                  value1 = dfunc[0]*distance_versor;
                  Vector value2_mol1_vector1 = dfunc[1]*der_cosTheta_mol1_vector1;
                  Vector value2_mol1_vector2 = dfunc[1]*der_cosTheta_mol1_vector2;
                  Vector value2_mol2_vector1 = dfunc[1]*der_cosTheta_mol2_vector1;
                  Vector value2_mol2_vector2 = dfunc[1]*der_cosTheta_mol2_vector2;
                    
                  gofrPrimeCenter[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                  gofrPrimeStart1[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector1;
                  gofrPrimeEnd1[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector1;
                  gofrPrimeStart2[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector2;
                  gofrPrimeEnd2[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector2;
             
                  gofrPrimeCenter[neighbor*nhist1_nhist2_+k*nhist_[1]+h] -= value1;
                  gofrPrimeStart1[neighbor*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol2_vector1;
                  gofrPrimeEnd1[neighbor*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol2_vector1;
                  gofrPrimeStart2[neighbor*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol2_vector2;
                  gofrPrimeEnd2[neighbor*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol2_vector2;
             
                  Tensor vv1(value1, distance);
                  Tensor vv2_mol1_vector1(value2_mol1_vector1, mol1_vector1);
                  Tensor vv2_mol1_vector2(value2_mol1_vector2, mol1_vector2);
                  Tensor vv2_mol2_vector1(value2_mol2_vector1, mol2_vector1);
                  Tensor vv2_mol2_vector2(value2_mol2_vector2, mol2_vector2);
             
                  gofrVirial[k][h] += vv1 + vv2_mol1_vector1 + vv2_mol1_vector2+ vv2_mol2_vector1 + vv2_mol2_vector2;
               }
            }
          }
        }
     }
  } else if (!doneigh && doLowComm) {
    for(unsigned int i=rank;i<center_lista.size();i+=stride) {
      unsigned atom1_mol1=i+center_lista.size();
      unsigned atom2_mol1=i+2*center_lista.size();
      unsigned atom3_mol1=i+3*center_lista.size();
      unsigned atom4_mol1=i+4*center_lista.size();

      Vector mol1_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
      Vector mol1_vector2=pbcDistance(getPosition(atom3_mol1),getPosition(atom4_mol1)); 
      Vector mol_vector1=crossProduct(mol1_vector1,mol1_vector2); 
      double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
      double inv_v1=1./norm_v1;
      double inv_v1_sqr=inv_v1*inv_v1;

      for(unsigned int j=0;j<center_lista.size();j+=1) {
        double d2;
        Vector distance;
        Vector distance_versor;
        if(getAbsoluteIndex(i)==getAbsoluteIndex(j)) continue;
        if(pbc){
         distance=pbcDistance(getPosition(i),getPosition(j));
        } else {
         distance=delta(getPosition(i),getPosition(j));
        }
        if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
          double distanceModulo=std::sqrt(d2);
          Vector distance_versor = distance / distanceModulo;
          unsigned bin=std::floor(distanceModulo/deltar);

          unsigned atom1_mol2=j+center_lista.size();
          unsigned atom2_mol2=j+2*center_lista.size();
          unsigned atom3_mol2=j+3*center_lista.size();
          unsigned atom4_mol2=j+4*center_lista.size();

          Vector mol2_vector1=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2)); 
          Vector mol2_vector2=pbcDistance(getPosition(atom3_mol2),getPosition(atom4_mol2)); 
          Vector mol_vector2=crossProduct(mol2_vector1,mol2_vector2); 
          double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);

          double inv_v2=1./norm_v2;
          double inv_v1_inv_v2=inv_v1*inv_v2;
          double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;

          Vector der_cosTheta_vector1;
          der_cosTheta_vector1[0]=(-mol1_vector2[2]*mol_vector2[1]+mol1_vector2[1]*mol_vector2[2])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*(-mol1_vector2[2]*mol_vector1[1]+mol1_vector2[1]*mol_vector1[2]);
          der_cosTheta_vector1[1]=( mol1_vector2[2]*mol_vector2[0]-mol1_vector2[0]*mol_vector2[2])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*( mol1_vector2[2]*mol_vector1[0]-mol1_vector2[0]*mol_vector1[2]);
          der_cosTheta_vector1[2]=(-mol1_vector2[1]*mol_vector2[0]+mol1_vector2[0]*mol_vector2[1])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*(-mol1_vector2[1]*mol_vector1[0]+mol1_vector2[0]*mol_vector1[1]);

          Vector der_cosTheta_vector2;
          der_cosTheta_vector2[0]=( mol1_vector1[2]*mol_vector2[1]-mol1_vector1[1]*mol_vector2[2])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*( mol1_vector1[2]*mol_vector1[1]-mol1_vector1[1]*mol_vector1[2]);
          der_cosTheta_vector2[1]=(-mol1_vector1[2]*mol_vector2[0]+mol1_vector1[0]*mol_vector2[2])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*(-mol1_vector1[2]*mol_vector1[0]+mol1_vector1[0]*mol_vector1[2]);
          der_cosTheta_vector2[2]=( mol1_vector1[1]*mol_vector2[0]-mol1_vector1[0]*mol_vector2[1])*inv_v1_inv_v2 - cosAngle*inv_v1_sqr*( mol1_vector1[1]*mol_vector1[0]-mol1_vector1[0]*mol_vector1[1]);

          if (doUpDownSymmetry && cosAngle<0) {
             der_cosTheta_vector1 *= -1.;
             der_cosTheta_vector2 *= -1.;
          }

          unsigned binAngle;
          if (doUpDownSymmetry && cosAngle<0) {
             binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
          } else {
             binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
          }
          int minBin, maxBin; // These cannot be unsigned
          // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
          minBin=bin - deltaBin;
          if (minBin < 0) minBin=0;
          if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
          maxBin=bin +  deltaBin;
          if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
          int minBinAngle, maxBinAngle; // These cannot be unsigned
          // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
          minBinAngle=binAngle - deltaBinAngle;
          maxBinAngle=binAngle +  deltaBinAngle;
          for(int k=minBin;k<maxBin+1;k+=1) {
            double invNormKernel=invNormConstantBase/x1sqr[k];
            vector<double> pos(2);
            pos[0]=x1[k]-distanceModulo;
            for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
               double theta=startCosAngle+deltaCosAngle*l;
               if (doUpDownSymmetry && cosAngle<0) {
                  pos[1]=theta+cosAngle;
               } else {
                  pos[1]=theta-cosAngle;
               }
               // Include periodic effects
               int h;
               if (l<0) {
                  h=-l;
               } else if (l>(nhist_[1]-1)) {
                  h=2*nhist_[1]-l-2;
               } else {
                  h=l;
               }
               Vector value1;
               Vector value2_mol1_vector1;
               Vector value2_mol1_vector2;
               vector<double> dfunc(2);
               if (l==(nhist_[1]-1) || l==0) {
                  gofr[k][h] += kernel(pos,2*invNormKernel,dfunc)/2.;
               } else {
                  gofr[k][h] += kernel(pos,invNormKernel,dfunc)/2.;
               }
               value1 = dfunc[0]*distance_versor;
               value2_mol1_vector1 = dfunc[1]*der_cosTheta_vector1;
               value2_mol1_vector2 = dfunc[1]*der_cosTheta_vector2;

               gofrPrimeCenter[i*nhist1_nhist2_+k*nhist_[1]+h] += value1;
               gofrPrimeStart1[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector1;
               gofrPrimeEnd1[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector1;
               gofrPrimeStart2[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector2;
               gofrPrimeEnd2[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector2;

               Tensor vv1(value1, distance);
               Tensor vv2_mol1_vector1(value2_mol1_vector1, mol1_vector1);
               Tensor vv2_mol1_vector2(value2_mol1_vector2, mol1_vector2);
               gofrVirial[k][h] += vv1/2. + vv2_mol1_vector1 + vv2_mol1_vector2;
            }
          }
        }
      }
    }
  } else if (!doneigh && !doLowComm) {
    for(unsigned int i=rank;i<(center_lista.size()-1);i+=stride) {
      unsigned atom1_mol1=i+center_lista.size();
      unsigned atom2_mol1=i+2*center_lista.size();
      unsigned atom3_mol1=i+3*center_lista.size();
      unsigned atom4_mol1=i+4*center_lista.size();

      Vector mol1_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
      Vector mol1_vector2=pbcDistance(getPosition(atom3_mol1),getPosition(atom4_mol1)); 
      Vector mol_vector1=crossProduct(mol1_vector1,mol1_vector2); 
      double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
      double inv_v1=1./norm_v1;
      double inv_v1_sqr=inv_v1*inv_v1;

      for(unsigned int j=i+1;j<center_lista.size();j+=1) {
        double d2;
        Vector distance;
        Vector distance_versor;
        if(getAbsoluteIndex(i)==getAbsoluteIndex(j)) continue;
        if(pbc){
         distance=pbcDistance(getPosition(i),getPosition(j));
        } else {
         distance=delta(getPosition(i),getPosition(j));
        }
        if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
          double distanceModulo=std::sqrt(d2);
          Vector distance_versor = distance / distanceModulo;
          unsigned bin=std::floor(distanceModulo/deltar);

          unsigned atom1_mol2=j+center_lista.size();
          unsigned atom2_mol2=j+2*center_lista.size();
          unsigned atom3_mol2=j+3*center_lista.size();
          unsigned atom4_mol2=j+4*center_lista.size();

          Vector mol2_vector1=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2)); 
          Vector mol2_vector2=pbcDistance(getPosition(atom3_mol2),getPosition(atom4_mol2)); 
          Vector mol_vector2=crossProduct(mol2_vector1,mol2_vector2); 
          double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);

          double inv_v2=1./norm_v2;
          double inv_v2_sqr=inv_v2*inv_v2;
          double inv_v1_inv_v2=inv_v1*inv_v2;
          double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;

          Vector der_cosTheta_mol1_vector1;
          der_cosTheta_mol1_vector1[0]=(-mol1_vector2[2]*mol_vector2[1]+mol1_vector2[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[2]*mol_vector1[1]+mol1_vector2[1]*mol_vector1[2]);
          der_cosTheta_mol1_vector1[1]=( mol1_vector2[2]*mol_vector2[0]-mol1_vector2[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector2[2]*mol_vector1[0]-mol1_vector2[0]*mol_vector1[2]);
          der_cosTheta_mol1_vector1[2]=(-mol1_vector2[1]*mol_vector2[0]+mol1_vector2[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector2[1]*mol_vector1[0]+mol1_vector2[0]*mol_vector1[1]);

          Vector der_cosTheta_mol1_vector2;
          der_cosTheta_mol1_vector2[0]=( mol1_vector1[2]*mol_vector2[1]-mol1_vector1[1]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[2]*mol_vector1[1]-mol1_vector1[1]*mol_vector1[2]);
          der_cosTheta_mol1_vector2[1]=(-mol1_vector1[2]*mol_vector2[0]+mol1_vector1[0]*mol_vector2[2])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*(-mol1_vector1[2]*mol_vector1[0]+mol1_vector1[0]*mol_vector1[2]);
          der_cosTheta_mol1_vector2[2]=( mol1_vector1[1]*mol_vector2[0]-mol1_vector1[0]*mol_vector2[1])*inv_v1_inv_v2 -cosAngle*inv_v1_sqr*( mol1_vector1[1]*mol_vector1[0]-mol1_vector1[0]*mol_vector1[1]);

          Vector der_cosTheta_mol2_vector1;
          der_cosTheta_mol2_vector1[0]=(-mol2_vector2[2]*mol_vector1[1]+mol2_vector2[1]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector2[2]*mol_vector2[1]+mol2_vector2[1]*mol_vector2[2]);
          der_cosTheta_mol2_vector1[1]=( mol2_vector2[2]*mol_vector1[0]-mol2_vector2[0]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector2[2]*mol_vector2[0]-mol2_vector2[0]*mol_vector2[2]);
          der_cosTheta_mol2_vector1[2]=(-mol2_vector2[1]*mol_vector1[0]+mol2_vector2[0]*mol_vector1[1])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector2[1]*mol_vector2[0]+mol2_vector2[0]*mol_vector2[1]);

          Vector der_cosTheta_mol2_vector2;
          der_cosTheta_mol2_vector2[0]=( mol2_vector1[2]*mol_vector1[1]-mol2_vector1[1]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector1[2]*mol_vector2[1]-mol2_vector1[1]*mol_vector2[2]);
          der_cosTheta_mol2_vector2[1]=(-mol2_vector1[2]*mol_vector1[0]+mol2_vector1[0]*mol_vector1[2])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*(-mol2_vector1[2]*mol_vector2[0]+mol2_vector1[0]*mol_vector2[2]);
          der_cosTheta_mol2_vector2[2]=( mol2_vector1[1]*mol_vector1[0]-mol2_vector1[0]*mol_vector1[1])*inv_v1_inv_v2 -cosAngle*inv_v2_sqr*( mol2_vector1[1]*mol_vector2[0]-mol2_vector1[0]*mol_vector2[1]);

          if (doUpDownSymmetry && cosAngle<0) {
             der_cosTheta_mol1_vector1 *= -1.;
             der_cosTheta_mol1_vector2 *= -1.;
             der_cosTheta_mol2_vector1 *= -1.;
             der_cosTheta_mol2_vector2 *= -1.;
          }

          unsigned binAngle;
          if (doUpDownSymmetry && cosAngle<0) {
             binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
          } else {
             binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
          }
          int minBin, maxBin; // These cannot be unsigned
          // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
          minBin=bin - deltaBin;
          if (minBin < 0) minBin=0;
          if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
          maxBin=bin +  deltaBin;
          if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
          int minBinAngle, maxBinAngle; // These cannot be unsigned
          // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
          minBinAngle=binAngle - deltaBinAngle;
          maxBinAngle=binAngle +  deltaBinAngle;
          for(int k=minBin;k<maxBin+1;k+=1) {
            double invNormKernel=invNormConstantBase/x1sqr[k];
            vector<double> pos(2);
            pos[0]=x1[k]-distanceModulo;
            for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
               double theta=startCosAngle+deltaCosAngle*l;
               if (doUpDownSymmetry && cosAngle<0) {
                  pos[1]=theta+cosAngle;
               } else {
                  pos[1]=theta-cosAngle;
               }
               // Include periodic effects
               int h;
               if (l<0) {
                  h=-l;
               } else if (l>(nhist_[1]-1)) {
                  h=2*nhist_[1]-l-2;
               } else {
                  h=l;
               }
               Vector value1;
               vector<double> dfunc(2);
               if (l==(nhist_[1]-1) || l==0) {
                  gofr[k][h] += kernel(pos,2*invNormKernel,dfunc);
               } else {
                  gofr[k][h] += kernel(pos,invNormKernel,dfunc);
               }
               value1 = dfunc[0]*distance_versor;
               Vector value2_mol1_vector1 = dfunc[1]*der_cosTheta_mol1_vector1;
               Vector value2_mol1_vector2 = dfunc[1]*der_cosTheta_mol1_vector2;
               Vector value2_mol2_vector1 = dfunc[1]*der_cosTheta_mol2_vector1;
               Vector value2_mol2_vector2 = dfunc[1]*der_cosTheta_mol2_vector2;
                 
               gofrPrimeCenter[i*nhist1_nhist2_+k*nhist_[1]+h] += value1;
               gofrPrimeStart1[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector1;
               gofrPrimeEnd1[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector1;
               gofrPrimeStart2[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1_vector2;
               gofrPrimeEnd2[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1_vector2;

               gofrPrimeCenter[j*nhist1_nhist2_+k*nhist_[1]+h] -= value1;
               gofrPrimeStart1[j*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol2_vector1;
               gofrPrimeEnd1[j*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol2_vector1;
               gofrPrimeStart2[j*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol2_vector2;
               gofrPrimeEnd2[j*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol2_vector2;

               Tensor vv1(value1, distance);
               Tensor vv2_mol1_vector1(value2_mol1_vector1, mol1_vector1);
               Tensor vv2_mol1_vector2(value2_mol1_vector2, mol1_vector2);
               Tensor vv2_mol2_vector1(value2_mol2_vector1, mol2_vector1);
               Tensor vv2_mol2_vector2(value2_mol2_vector2, mol2_vector2);

               gofrVirial[k][h] += vv1 + vv2_mol1_vector1 + vv2_mol1_vector2+ vv2_mol2_vector1 + vv2_mol2_vector2;
            }
          }
        }
      }
    }
  }
  //std::cout << "Main loop: " << float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  if(!serial){
    comm.Sum(gofr);
    if (!doNotCalculateDerivatives() ) {
       comm.Sum(gofrVirial);
       if (!doLowComm) {
          comm.Sum(gofrPrimeCenter);
          comm.Sum(gofrPrimeStart1);
          comm.Sum(gofrPrimeEnd1);
          comm.Sum(gofrPrimeStart2);
          comm.Sum(gofrPrimeEnd2);
       }
    }
  }
  //std::cout << "Communication: " <<  float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  if (doAverageGofr) {
     if (!doNotCalculateDerivatives()) error("Cannot calculate derivatives or bias using the AVERAGE_GOFR option");
     double factor;
     if (averageGofrTau==0 || iteration < averageGofrTau) {
        iteration += 1;
        factor = 1./( (double) iteration );
     } else factor = 2./((double) averageGofrTau + 1.);
     for(unsigned i=0;i<nhist_[0];++i){
        for(unsigned j=0;j<nhist_[1];++j){
           avgGofr[i][j] += (gofr[i][j]-avgGofr[i][j])*factor;
           gofr[i][j] = avgGofr[i][j];
        }
     }
  }
  // Output of gofr
  if (doOutputGofr && (getStep()%outputStride==0)) outputGofr(gofr,"gofr.txt");
  // Construct integrand
  Matrix<double> integrand(nhist_[0],nhist_[1]);
  Matrix<double> logGofrx1sqr(nhist_[0],nhist_[1]);
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        if (doReferenceGofr) {
           if (gofr[i][j]<1.e-10) {
              integrand[i][j] = (referenceGofr[i][j]+epsilon)*x1sqr[i];
           } else {
              logGofrx1sqr[i][j] = std::log(gofr[i][j]/(referenceGofr[i][j]+epsilon))*x1sqr[i];
              integrand[i][j] = gofr[i][j]*logGofrx1sqr[i][j]+(-gofr[i][j]+referenceGofr[i][j]+epsilon)*x1sqr[i];
           }
        } else {
           if (gofr[i][j]<1.e-10) {
              integrand[i][j] = x1sqr[i];
           } else {
              logGofrx1sqr[i][j] = std::log(gofr[i][j])*x1sqr[i];
              integrand[i][j] = gofr[i][j]*logGofrx1sqr[i][j]+(-gofr[i][j]+1)*x1sqr[i];
           }
        }
     }
  }
  vector<double> delta(2);
  delta[0]=deltar;
  delta[1]=deltaCosAngle;
  double TwoPiDensityVolAngles;
  if (densityReference>0) TwoPiDensityVolAngles=(2*pi/volumeOfAngles)*densityReference;
  else TwoPiDensityVolAngles=(2*pi/volumeOfAngles)*density;
  double pairEntropy=-TwoPiDensityVolAngles*integrate(integrand,delta);
  //std::cout << "Integrand and integration: " << float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  // Derivatives
  vector<Vector> deriv(getNumberOfAtoms());
  Tensor virial;
  if (!doNotCalculateDerivatives() ) {
    if (doneigh) {
       for(unsigned int k=0;k<nl->getNumberOfLocalAtoms();k+=1) {
         unsigned index=nl->getIndexOfLocalAtom(k);
         // Center atom
         unsigned start1_atom=index+center_lista.size();
         unsigned end1_atom=index+2*center_lista.size();
         unsigned start2_atom=index+3*center_lista.size();
         unsigned end2_atom=index+4*center_lista.size();
         Matrix<Vector> integrandDerivatives(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStart1(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEnd1(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStart2(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEnd2(nhist_[0],nhist_[1]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofr[i][j]>1.e-10) {
               integrandDerivatives[i][j] = gofrPrimeCenter[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesStart1[i][j] = gofrPrimeStart1[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesEnd1[i][j] = gofrPrimeEnd1[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesStart2[i][j] = gofrPrimeStart2[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesEnd2[i][j] = gofrPrimeEnd2[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
             }
           }
         }
         deriv[index] = -TwoPiDensityVolAngles*integrate(integrandDerivatives,delta);
         deriv[start1_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesStart1,delta);
         deriv[end1_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesEnd1,delta);
         deriv[start2_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesStart2,delta);
         deriv[end2_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesEnd2,delta);
       }
    } else {
       for(unsigned int k=rank;k<center_lista.size();k+=stride) {
         // Center atom
         unsigned start1_atom=k+center_lista.size();
         unsigned end1_atom=k+2*center_lista.size();
         unsigned start2_atom=k+3*center_lista.size();
         unsigned end2_atom=k+4*center_lista.size();
         Matrix<Vector> integrandDerivatives(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStart1(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEnd1(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStart2(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEnd2(nhist_[0],nhist_[1]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofr[i][j]>1.e-10) {
               integrandDerivatives[i][j] = gofrPrimeCenter[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesStart1[i][j] = gofrPrimeStart1[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesEnd1[i][j] = gofrPrimeEnd1[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesStart2[i][j] = gofrPrimeStart2[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
               integrandDerivativesEnd2[i][j] = gofrPrimeEnd2[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
             }
           }
         }
         deriv[k] = -TwoPiDensityVolAngles*integrate(integrandDerivatives,delta);
         deriv[start1_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesStart1,delta);
         deriv[end1_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesEnd1,delta);
         deriv[start2_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesStart2,delta);
         deriv[end2_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesEnd2,delta);
       }
    }
    if(!serial){
      comm.Sum(&deriv[0][0],3*getNumberOfAtoms());
    }
    // Virial of positions
    // Construct virial integrand
    Matrix<Tensor> integrandVirial(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          if (gofr[i][j]>1.e-10) {
             integrandVirial[i][j] = gofrVirial[i][j]*logGofrx1sqr[i][j];
          }
      }
    }
    // Integrate virial
    virial = -TwoPiDensityVolAngles*integrate(integrandVirial,delta);
    // Virial of volume
    // Construct virial integrand
    Matrix<double> integrandVirialVolume(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          if (densityReference>0) {
            integrandVirialVolume[i][j] = -gofr[i][j]*logGofrx1sqr[i][j];
          } else if (doReferenceGofr && densityReference<0) {
            integrandVirialVolume[i][j] = (-gofr[i][j]+referenceGofr[i][j]+epsilon)*x1sqr[i];
          } else {
            integrandVirialVolume[i][j] = (-gofr[i][j]+1)*x1sqr[i];
          }
       }
    }
    // Integrate virial
    virial += -TwoPiDensityVolAngles*integrate(integrandVirialVolume,delta)*Tensor::identity();
  }
  //std::cout << "Derivatives integration: " << float( clock () - begin_time ) << "\n";
  // Assign output quantities
  for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
  setValue           (pairEntropy);
  setBoxDerivatives  (virial);
}

double PairOrientationalEntropyPerpendicular::kernel(vector<double> distance, double invNormKernel, vector<double>&der)const{
  // Gaussian function and derivative
  double result = invNormKernel*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  //double result = invTwoPiSigma1Sigma2*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  der[0] = -distance[0]*result/sigma1Sqr;
  der[1] = -distance[1]*result/sigma2Sqr;
  return result;
}

double PairOrientationalEntropyPerpendicular::integrate(Matrix<double> integrand, vector<double> delta)const{
  // Trapezoid rule
  double result = 0.;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

Vector PairOrientationalEntropyPerpendicular::integrate(Matrix<Vector> integrand, vector<double> delta)const{
  // Trapezoid rule
  Vector result;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

Tensor PairOrientationalEntropyPerpendicular::integrate(Matrix<Tensor> integrand, vector<double> delta)const{
  // Trapezoid rule
  Tensor result;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

void PairOrientationalEntropyPerpendicular::outputGofr(Matrix<double> gofr, const char* fileName) {
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        gofrOfile.printField("r",x1[i]).printField("theta",x2[j]).printField("gofr",gofr[i][j]).printField();
     }
     gofrOfile.printf("\n");
  }
  gofrOfile.printf("\n");
  gofrOfile.printf("\n");
}

}
}
