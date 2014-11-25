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

		Scene scene(folder + name + ".xml");

		JPEGImage img(scene.filename());
		int s_height = (int) scene.height();
		int s_width = (int) scene.width();
		int i_width = (int) img.width();
		int i_height = (int) img.height();

		if(i_height == 0 || i_width == 0) {
			std::cout << name << std::endl;
			std::cout << i_height << "," << i_width << std::endl;
		}
	}
	return 0;
}
