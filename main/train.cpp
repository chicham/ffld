//--------------------------------------------------------------------------------------------------
// Implementation of the papers "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012 and "Deformable Part Models with Individual Part Scaling",
// 24th British Machine Vision Conference, 2013.
//
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
//
// FFLDv2 is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License version 3 as published by the Free Software Foundation.
//
// FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
// General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with FFLDv2. If
// not, see <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "SimpleOpt.h"

#include "Mixture.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace FFLD;
using namespace std;
using namespace boost::filesystem;

namespace logging = boost::log;
namespace timing = boost::posix_time;

// SimpleOpt array of valid options
enum
{
	OPT_C, OPT_DATAMINE, OPT_INTERVAL, OPT_HELP, OPT_J, OPT_RELABEL, OPT_MODEL, OPT_NAME,
	OPT_PADDING, OPT_RESULT, OPT_SEED, OPT_OVERLAP, OPT_NB_COMP, OPT_NB_NEG
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_C, "-c", SO_REQ_SEP },
	{ OPT_C, "--C", SO_REQ_SEP },
	{ OPT_DATAMINE, "-d", SO_REQ_SEP },
	{ OPT_DATAMINE, "--datamine", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_J, "-j", SO_REQ_SEP },
	{ OPT_J, "--J", SO_REQ_SEP },
	{ OPT_RELABEL, "-l", SO_REQ_SEP },
	{ OPT_RELABEL, "--relabel", SO_REQ_SEP },
	{ OPT_MODEL, "-m", SO_REQ_SEP },
	{ OPT_MODEL, "--model", SO_REQ_SEP },
	{ OPT_NAME, "-n", SO_REQ_SEP },
	{ OPT_NAME, "--name", SO_REQ_SEP },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_RESULT, "-r", SO_REQ_SEP },
	{ OPT_RESULT, "--result", SO_REQ_SEP },
	{ OPT_SEED, "-s", SO_REQ_SEP },
	{ OPT_SEED, "--seed", SO_REQ_SEP },
	{ OPT_OVERLAP, "-v", SO_REQ_SEP },
	{ OPT_OVERLAP, "--overlap", SO_REQ_SEP },
	{ OPT_NB_COMP, "-x", SO_REQ_SEP },
	{ OPT_NB_COMP, "--nb-components", SO_REQ_SEP },
	{ OPT_NB_NEG, "-z", SO_REQ_SEP },
	{ OPT_NB_NEG, "--nb-negatives", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage()
{
	cout << "Usage: train [options] image_set.txt\n\n"
			"Options:\n"
			"  -c,--C <arg>             SVM regularization constant (default 0.002)\n"
			"  -d,--datamine <arg>      Maximum number of data-mining iterations within each "
			"training iteration  (default 10)\n"
			"  -e,--interval <arg>      Number of levels per octave in the HOG pyramid (default 5)"
			"\n"
			"  -h,--help                Display this information\n"
			"  -j,--J <arg>             SVM positive regularization constant boost (default 2)\n"
			"  -l,--relabel <arg>       Maximum number of training iterations (default 8, half if "
			"no part)\n"
			"  -m,--model <file>        Read the initial model from <file> (default zero model)\n"
			"  -n,--name <arg>          Name of the object to detect (default \"person\")\n"
			"  -p,--padding <arg>       Amount of zero padding in HOG cells (default 6)\n"
			"  -r,--result <file>       Write the trained model to <file> (default \"model.txt\")\n"
			"  -s,--seed <arg>          Random seed (default time(NULL))\n"
			"  -v,--overlap <arg>       Minimum overlap in latent positive search (default 0.7)\n"
			"  -x,--nb-components <arg> Number of mixture components (without symmetry, default 3)\n"
			"  -z,--nb-negatives <arg>  Maximum number of negative images to consider (default all)"
		 << endl;
}

// Train a mixture model
int main(int argc, char * argv[])
{
	// Default parameters
	double C = 0.002;
	int nbDatamine = 10;
	int interval = 5;
	double J = 2.0;
	int nbRelabel = 8;
	string model;
	Object::Name name = Object::PERSON;
	int padding = 6;
	string result("model.txt");
	int seed = static_cast<int>(time(0));
	double overlap = 0.7;
	int nbComponents = 3;
	int nbNegativeScenes = -1;
	
	// Parse the parameters
	CSimpleOpt args(argc, argv, SOptions);

	logging::add_console_log();
	path train_log;
	
	while (args.Next()) {
		if (args.LastError() == SO_SUCCESS) {
			if (args.OptionId() == OPT_RESULT) {
				result = args.OptionArg();
				train_log = path(result);
				timing::ptime now(timing::second_clock::universal_time());
				logging::add_file_log( train_log.stem().string() + "_train" + to_iso_extended_string(now) + ".log" );
			}
			else if (args.OptionId() == OPT_C) {
				C = atof(args.OptionArg());
				
				if (C <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal)<< "Invalid C arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_DATAMINE) {
				nbDatamine = atoi(args.OptionArg());
				
				if (nbDatamine <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid datamine arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_INTERVAL) {
				interval = atoi(args.OptionArg());
				
				if (interval <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid interval arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_HELP) {
				showUsage();
				return 0;
			}
			else if (args.OptionId() == OPT_J) {
				J = atof(args.OptionArg());
				
				if (J <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid J arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_RELABEL) {
				nbRelabel = atoi(args.OptionArg());
				
				if (nbRelabel <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid relabel arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_MODEL) {
				model = args.OptionArg();
			}
			else if (args.OptionId() == OPT_NAME) {
				string arg = args.OptionArg();
				transform(arg.begin(), arg.end(), arg.begin(), static_cast<int (*)(int)>(tolower));
				
// Redefine names with categories
// get number of categories from Names
				// const string Names[LEN] =
				// {
				// 	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
				// 	"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
				// 	"sheep", "sofa", "train", "tvmonitor", ...
				// };
				const string Names[80] =
				{
					"airplane", "apple", "backpack", "banana", "baseball bat",
              "baseball glove", "bear", "bed", "bench", "bicycle", "bird",
              "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake",
              "car", "carrot", "cat", "cell phone", "chair", "clock", "couch",
              "cow", "cup", "dining table", "dog", "donut", "elephant",
              "fire hydrant", "fork", "frisbee", "giraffe", "hair drier",
              "handbag", "horse", "hot_dog", "keyboard", "kite", "knife",
              "laptop", "microwave", "motorcycle", "mouse", "orange",
              "oven", "parking meter", "person", "pizza", "potted plant",
              "refrigerator", "remote", "sandwich", "scissors", "sheep",
              "sink", "skateboard", "skis", "snowboard", "spoon", "sports ball",
              "stop sign", "suitcase", "surfboard", "teddy bear", "tennis racket",
              "tie", "toaster", "toilet", "toothbrush", "traffic light", "train",
              "truck", "tv", "umbrella", "vase", "wine", "zebra"
				};
				
				const string * iter = find(Names, Names + 80, arg);
				
				if (iter == Names + 80) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid name arg " << args.OptionArg();
					return -1;
				}
				
				name = static_cast<Object::Name>(iter - Names);
			}
			else if (args.OptionId() == OPT_PADDING) {
				padding = atoi(args.OptionArg());
				
				if (padding <= 1) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid padding arg " << args.OptionArg();
					return -1;
				}
			}
			
			else if (args.OptionId() == OPT_SEED) {
				seed = atoi(args.OptionArg());
			}
			else if (args.OptionId() == OPT_OVERLAP) {
				overlap = atof(args.OptionArg());
				
				if ((overlap <= 0.0) || (overlap >= 1.0)) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid overlap arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_NB_COMP) {
				nbComponents = atoi(args.OptionArg());
				
				if (nbComponents <= 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid nb-components arg " << args.OptionArg();
					return -1;
				}
			}
			else if (args.OptionId() == OPT_NB_NEG) {
				nbNegativeScenes = atoi(args.OptionArg());
				
				if (nbNegativeScenes < 0) {
					showUsage();
					BOOST_LOG_TRIVIAL(fatal) << "Invalid nb-negatives arg " << args.OptionArg();
					return -1;
				}
			}
		}
		else {
			showUsage();
			BOOST_LOG_TRIVIAL(fatal) << "Unknown option " << args.OptionText();
			return -1;
		}
	}
	
	srand(seed);
	srand48(seed);

	
	if (!args.FileCount()) {
		showUsage();
		BOOST_LOG_TRIVIAL(fatal) << "No dataset provided";
		return -1;
	}
	else if (args.FileCount() > 1) {
		showUsage();
		BOOST_LOG_TRIVIAL(fatal) << "More than one dataset provided";
		return -1;
	}
	
	// Open the image set file
	const string file(args.File(0));
	BOOST_LOG_TRIVIAL(info) << "ImageSet: " << file;
	const size_t lastDot = file.find_last_of('.');
	
	if ((lastDot == string::npos) || (file.substr(lastDot) != ".txt")) {
		showUsage();
		cerr << "\nInvalid image set file " << file << ", should be .txt";
		return -1;
	}
	
	ifstream in(file.c_str());
	
	if (!in.is_open()) {
		showUsage();
		cerr << "\nInvalid image set file " << file;
		return -1;
	}
	
	// Find the annotations' folder (not sure that will work under Windows)
	const string folder = file.substr(0, file.find_last_of("/\\")) + "/../../Annotations/";
	
	// Load all the scenes
	int maxRows = 0;
	int maxCols = 0;
	int nbPositives = 0;
	int nbNegatives = 0;
	
	vector<Scene> scenes;
	
	while (in) {
		string line;
		getline(in, line);
		
		// Skip empty lines
		if (line.size() < 3){
			BOOST_LOG_TRIVIAL(warning) << "Empty line";
			continue;
		}
		
		// Check whether the scene is positive or negative
		const Scene scene(folder + line.substr(0, line.find(' ')) + ".xml");
		
		if (scene.empty())
			continue;
		
		bool positive = false;
		bool negative = true;
		
		for (int i = 0; i < scene.objects().size(); ++i) {
			if (scene.objects()[i].name() == name) {
				negative = false;
				
				if (!scene.objects()[i].difficult()){
					positive = true;
					nbPositives++;
				}
			}
			else
				nbNegatives++;
		}

		
		if (positive || (negative && nbNegativeScenes)) {
			scenes.push_back(scene);
			
			maxRows = max(maxRows, (scene.height() + 3) / 4 + padding);
			maxCols = max(maxCols, (scene.width() + 3) / 4 + padding);
			
			if (negative)
				--nbNegativeScenes;
		}
	}

	BOOST_LOG_TRIVIAL(info) << nbPositives << " positive samples";
	BOOST_LOG_TRIVIAL(info) << nbNegatives << " negative samples";
	
	if (scenes.empty()) {
		showUsage();
		BOOST_LOG_TRIVIAL(fatal) << "Invalid image_set file " << file;
		return -1;
	}
	
	// Initialize the Patchwork class
	if (!Patchwork::InitFFTW((maxRows + 15) & ~15, (maxCols + 15) & ~15)) {
		BOOST_LOG_TRIVIAL(fatal)<< "Error initializing the FFTW library";
		return - 1;
	}
	
	// The mixture to train
	Mixture mixture(nbComponents, scenes, name);

	if (mixture.empty()) {
		BOOST_LOG_TRIVIAL(fatal)<< "Error initializing the mixture model";
		return -1;
	}

	
	// Try to open the mixture
	if (!model.empty()) {
		ifstream in(model.c_str(), ios::binary);
		
		if (!in.is_open()) {
			showUsage();
			BOOST_LOG_TRIVIAL(fatal)<< "Invalid model file " << model;
			return -1;
		}
		
		in >> mixture;
		
		if (mixture.empty()) {
			showUsage();
			BOOST_LOG_TRIVIAL(fatal)<< "Invalid model file " << model;
			return -1;
		}
	}
	
	BOOST_LOG_TRIVIAL(info) << "Number of negative samples for train " << 5*nbPositives;
	if (model.empty())
		// mixture.train(scenes, name, padding, padding, interval, nbRelabel / 2, nbDatamine, 5*nbPositives, C, J, overlap);
		mixture.train(scenes, name, padding, padding, interval, nbRelabel / 2, nbDatamine, 24000, C, J, overlap);
	
	if (mixture.models()[0].parts().size() == 1)
		mixture.initializeParts(8, make_pair(6, 6));
	
	// mixture.train(scenes, name, padding, padding, interval, nbRelabel, nbDatamine, 5*nbPositives, C, J, overlap);
	mixture.train(scenes, name, padding, padding, interval, nbRelabel, nbDatamine, 24000, C, J, overlap);
	
	// Try to open the result file
	ofstream out(result.c_str(), ios::binary);
	
	if (!out.is_open()) {
		showUsage();
		BOOST_LOG_TRIVIAL(fatal) << "Invalid result file " << result;
		cout << mixture << endl; // Print the mixture as a last resort
		return -1;
	}
	
	out << mixture;
	
	return EXIT_SUCCESS;
}
