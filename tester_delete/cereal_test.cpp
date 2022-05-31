#include <iostream>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <iostream>
#include <sstream>

using namespace std;
struct MyClass 
{
  int x, y, z;
  bool saver;
  template<class Archive>
  void save(Archive & archive) const
  {
    if(saver){
        cout<<"this branch taken"<<endl;
        archive( x, y, z );
    }
    else{
        archive(x,y);
    }
     
  }

  template<class Archive>
  void load(Archive & archive)
  {
    if(saver){
        cout<<"this branch taken"<<endl;
        archive( x, y, z );
    }
    else{
        archive(x,y);
    } 
  }
};

struct newClass 
{
  int x, y, z;

  template<class Archive>
  void save(Archive & archive) const
  {
        // cout<<"this branch taken";
        archive( x, y, z );
     
  }

  template<class Archive>
  void load(Archive & archive)
  {
        // cout<<"this branch taken";
        archive( x, y, z );
    } 
  
};
int main(){
  std::stringstream ss1,ss2;

  MyClass m;
  m.x=1;m.y=2;m.z=3;m.saver=false;

  MyClass n;
  n.x=-1;n.y=-2;n.z=-3;n.saver=true;

  cereal::BinaryOutputArchive oarchive( ss1 );
  oarchive(m);
  // std::cout<<ss1.str()<<std::endl;
  
  cereal::BinaryOutputArchive oarchive2( ss2 );
  oarchive2(n);
  // std::cout<<ss2.str();

  MyClass m2;
  m2.saver=true;
  m2.z=100;
  m2.x=99;
  cereal::BinaryInputArchive iarchive(ss1);
  iarchive(m2);
  cout<<m2.x<<" "<<m2.y<<" "<<m2.z<<" "<<endl;

  // newClass m;
  // m.x=1;m.y=2;m.z=3;

  // newClass n;
  // n.x=-1;n.y=-2;n.z=-3;

  // cereal::BinaryOutputArchive oarchive( ss1 );
  // oarchive(m);
  // // std::cout<<ss1.str()<<std::endl;
  
  // cereal::BinaryOutputArchive oarchive2( ss2 );
  // oarchive2(n);
  // // std::cout<<ss2.str();

  // newClass m2;
  // m2.z=100;
  // m2.x=99;
  // cereal::BinaryInputArchive iarchive(ss1);
  // iarchive(m2);
  
  // cout<<m2.x<<" "<<m2.y<<" "<<m2.z<<" "<<endl;
}