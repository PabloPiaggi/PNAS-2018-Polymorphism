/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2017 The plumed team
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
#ifndef __PLUMED_tools_NeighborListParallel_h
#define __PLUMED_tools_NeighborListParallel_h

#include "Vector.h"
#include "AtomNumber.h"
#include "Communicator.h"
#include "Log.h"

#include <vector>

namespace PLMD {

class Pbc;

/// \ingroup TOOLBOX
/// A class that implements neighbor lists from two lists or a single list of atoms
class NeighborListParallel
{
  bool do_pair_,do_pbc_,twolists_,firsttime_;
/// Choose between half and full list
  bool do_full_list_;
  const PLMD::Pbc* pbc_;
/// Full list of atoms involved
  std::vector<PLMD::AtomNumber> fullatomlist_,requestlist_;
/// Positions during the building of the NL
/// They are used for checking if the next build is dangerous
  std::vector<PLMD::Vector> positions_old_;
/// Neighbor list
  std::vector<std::vector<unsigned>> neighbors_;
  //std::vector<std::pair<unsigned,unsigned> > neighbors_;
/// List of the "local" atoms each thread has
  std::vector<unsigned> local_atoms_;
  double distance_, skin_;
  int stride_;
  unsigned nlist0_,nlist1_,nallpairs_,lastupdate_;
  unsigned dangerousBuilds_, numberOfBuilds_;
  double maxLoadImbalance_,avgLoadImbalance_, avgTotalNeighbors_;
/// Return the pair of indexes in the positions array
/// of the two atoms forming the i-th pair among all possible pairs
  //std::pair<unsigned,unsigned> getIndexPair(unsigned i);
/// Communicator
  Communicator& mycomm;
/// Log
  Log& mylog;
/// MPI stuff
  unsigned mpi_rank, mpi_stride;
/// Linked list stuff
  double binsizex, binsizey, binsizez, bininvx, bininvy, bininvz ;
  bool do_linked_list_;
/// Number of bins in x,y,z
  int nbinx, nbiny, nbinz;
/// Head of chain for each bin
  std::vector<int> binhead;
/// Linked list for each bin
  std::vector<unsigned> bins;
/// sx,sy,sz = max range of stencil in each dim
  int sx,sy,sz;
/// Number of stencils
  unsigned nstencil;
/// Stencil indices
  std::vector<unsigned> stencilx, stencily, stencilz;
public:
  NeighborListParallel(const std::vector<PLMD::AtomNumber>& list0,
               const std::vector<PLMD::AtomNumber>& list1,
               const bool& do_pair, const bool& do_pbc, const PLMD::Pbc& pbc, Communicator& cc,
               Log& log, const double& distance=1.0e+30,  const bool& do_full_list=false, const int& stride=0, const double& skin=0.1);
  NeighborListParallel(const std::vector<PLMD::AtomNumber>& list0, const bool& do_pbc,
               const PLMD::Pbc& pbc, Communicator& cc, Log& log, const double& distance=1.0e+30,
               const bool& do_full_list=false, const int& stride=0, const double& skin=0.1);
/// Return the list of all atoms. These are needed to rebuild the neighbor list.
  std::vector<PLMD::AtomNumber>& getFullAtomList();
/// Check if the nieghbor list must be rebuilt
  bool isListStillGood(const std::vector<Vector>& positions);
/// Update the indexes in the neighbor list to match the
/// ordering in the new positions array
/// and return the new list of atoms that must be requested to the main code
//  std::vector<PLMD::AtomNumber>& getReducedAtomList();
/// Update the neighbor list and prepare the new
/// list of atoms that will be requested to the main code
  void update(const std::vector<PLMD::Vector>& positions);
/// Construct a half neighbor list, containing only i,j and not j,i
  void updateHalfList(const std::vector<Vector>& positions);
/// Construct a full neighbor list, containing both i,j and j,i
  void updateFullList(const std::vector<Vector>& positions);
/// Construct a full neighbor list, containing both i,j and j,i ; Linked lists are used
  void updateFullListWithLinkedList(const std::vector<Vector>& positions);
/// Construct a half neighbor list, containing only i,j and not j,i ; Linked lists are used
  void updateHalfListWithLinkedList(const std::vector<Vector>& positions);
/// Get the update stride of the neighbor list
  int getStride() const;
/// Get the last step in which the neighbor list was updated
  unsigned getLastUpdate() const;
/// Set the step of the last update
  void setLastUpdate(unsigned step);
/// Get the size of the neighbor list - Obsolete
  unsigned size() const;
/// Get the i-th pair of the neighbor list - Obsolete
  std::pair<unsigned,unsigned> getClosePair(unsigned i) const;
/// Get the list of neighbors of the i-th atom
  std::vector<unsigned> getNeighbors(unsigned i);
  ~NeighborListParallel() {}
/// Print statistics of neighbor list
  void printStats();
/// Gather statistics of neighbor list
  void gatherStats(const std::vector<PLMD::Vector>& positions);
/// Get the local number of atoms (the ones each thread is handling)
  unsigned getNumberOfLocalAtoms() const;
/// Get the index if the i-th local atom
  unsigned getIndexOfLocalAtom(unsigned i) const;
/// Prepare linked lists
  void prepareLinkedList();
/// Coordinate to bin
  unsigned coord2bin(const Vector& position);
/// Coordinate to bin, get also position in x,y,z
  unsigned coord2bin(const Vector& position, unsigned& ibinx, unsigned& ibiny, unsigned& ibinz);
/// Find the k-th neighboring cell of cell atombinx,atombiny,atombinz
  unsigned neighborCell(const unsigned& k, const unsigned& atombinx, const unsigned& atombiny, const unsigned& atombinz);
/// Calculate the distance with or without pbc, as defined with the do_pbc_ flag
  Vector distance(const Vector& position1, const Vector& position2);
};

}

#endif
