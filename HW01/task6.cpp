#include <iostream>
#include <string> 

int main(int argc, char *argv[]){
    if(argc == 1)
    {
        std::cout<<"Provide a valid number N\n";
        return -1;
    }
    
    int num = std::stoi(argv[1]);
    for(int i=0;i<=num;i++){
        std::cout<<i<<" ";
    }
    std::cout<<"\n";
    for(int i=num;i>=0;i--){
        std::cout<<i<<" ";
    }
    std::cout<<"\n";

}