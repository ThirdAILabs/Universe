#include <iostream>
#include <cereal/archives/json.hpp>
#include <iostream>

struct MyClass 
{
  int x, y, z;
  bool saver;
  template<class Archive>
  void save(Archive & archive) const
  {
    if(saver){
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
        archive( x, y, z );
    }
    else{
        archive(x,y);
    } 
  }
};

int main(){
  MyClass m;
  m.x=1;m.y=2;m.z=3;m.saver=false;
  MyClass n;
  n.x=-1;n.y=-2;n.z=-3;n.saver=true;
  cereal::JSONOutputArchive ar( std::cout );
  ar( CEREAL_NVP(m), CEREAL_NVP(n) );
}