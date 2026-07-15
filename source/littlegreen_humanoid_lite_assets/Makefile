
ROBOT ?= littlegreen_humanoid_lite

.PHONY: clean urdf mjcf usd

all: clean urdf mjcf usd

urdf:
	mkdir -p ./data/urdf/assets/
	cp ./data/scad/* ./data/urdf/assets/
	onshape-to-robot ./data/urdf/
	python postprocess.py --mode urdf --name ${ROBOT}

mjcf:
	mkdir -p ./data/mjcf/assets/
	cp ./data/scad/* ./data/mjcf/assets/
	onshape-to-robot ./data/mjcf/
	mv ./data/mjcf/scene.xml ./data/mjcf/${ROBOT}_scene.xml
	python postprocess.py --mode mjcf --name ${ROBOT}

usd:
	python ${ISAACLAB_HOME}/scripts/tools/convert_urdf.py ./data/urdf/${ROBOT}.urdf ./data/usd/${ROBOT}.usd --merge-joints

clean:
	rm -rf ./data/mjcf/assets/
	rm -rf ./data/urdf/assets/
	onshape-to-robot-clear-cache
