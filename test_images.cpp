#include "SimpleOpt.h"
#include "Scene.h"
#include "JPEGImage.h"
#include <iostream>
#include <fstream>

using namespace FFLD;
using namespace std;

int main(int argc, char *argv[])
{
	const string file(argv[1]);

	ifstream in(file.c_str(), ios::binary);
	const string folder = file.substr(0, file.find_last_of("/\\")) + "/../../Annotations/";

	int maxRows = 0;
	int maxCols = 0;
	int nbNegativeScenes = -1;
	int padding = 6;

	while(in){
		string line;
		getline(in, line);
		const string name = line.substr(0, line.find(' '));
		std::cout << name << std::endl;

		Scene scene(folder + name + ".xml");

		JPEGImage img(scene.filename());

		if(scene.height() != img.height() || scene.width()!=img.height()){
			std::cout << scene.height() << "," << scene.width() << std::endl;
			std::cout << img.height() << "," << img.width() << std::endl;
		}
	}
	return 0;
}
