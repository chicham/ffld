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
		std::cout << line << std::endl;

		Scene scene(folder + line.substr(0, line.find(' ')) + ".xml");
		std::cout << "Name: " << scene.filename() << std::endl;
		std::cout << "XML Width: " << scene.width() << std::endl;
		std::cout << "XML Height: " << scene.height() << std::endl;

		JPEGImage img(scene.filename());
		std::cout << "IMG Width: " << img.width() << std::endl;
		std::cout << "IMG Height: " << img.height() << std::endl;
		std::cout << "IMG Depth: " << img.depth() << std::endl;


	}

	return 0;
}
