/*
 *    Copyright (C) 2025 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"
#include <vector>
#include <cppitertools/groupby.hpp>
#include <cppitertools/range.hpp>
#include <time.h>
#include <cppitertools/enumerate.hpp>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
}


void SpecificWorker::initialize()
{
	std::cout << "Initialize worker" << std::endl;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		///////////// Your code ////////
		// Viewer
		viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
		auto [r, e] = viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
		robot_draw = r;
		//viewer->show();


		viewer_room = new AbstractGraphicViewer(this->frame_room, params.GRID_MAX_DIM);
		auto [rr, re] = viewer_room->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
		robot_room_draw = rr;
		// draw room in viewer_room
		viewer_room->scene.addRect(nominal_rooms[habitacion].rect(), QPen(Qt::black, 30));
		//viewer_room->show();
		show();


		// initialise robot pose
		robot_pose.setIdentity();
		robot_pose.translate(Eigen::Vector2d(0.0,0.0));


		// time series plotter for match error
		TimeSeriesPlotter::Config plotConfig;
		plotConfig.title = "Maximum Match Error Over Time";
		plotConfig.yAxisLabel = "Error (mm)";
		plotConfig.timeWindowSeconds = 15.0; // Show a 15-second window
		plotConfig.autoScaleY = true;       // We will set a fixed range
		plotConfig.yMin = 0;
		plotConfig.yMax = 1000;
		time_series_plotter = std::make_unique<TimeSeriesPlotter>(frame_plot_error, plotConfig);
		match_error_graph = time_series_plotter->addGraph("", Qt::blue);


		// stop robot
		//move_robot(0, 0, 0);
	}
}


RoboCompLidar3D::TPoints SpecificWorker::get_filtered_lidar_data()
{
	RoboCompLidar3D::TPoints filter_data;
	try
	{
		const auto data = lidar3d_proxy->getLidarDataWithThreshold2d("helios", 12000, 1);
		if (data.points.empty())
		{
			qWarning() << "No points received";
			return filter_data; // Return empty filter_data
		}

		filter_data = filter_min_distance(data.points);
		draw_lidar(filter_data, &viewer->scene);
	}
	catch (const Ice::Exception& e)
	{
		std::cout << e << " Conexión con Laser\n";
	}

	return filter_data;
}

void SpecificWorker::compute()
{
   RoboCompLidar3D::TPoints data = get_filtered_lidar_data();
   data = door_detector.filter_points(data, &viewer->scene);

   draw_lidar(data, &viewer->scene);

	if (localised) localise(data);

	auto [corners, lines] = room_detector.compute_corners(data, &viewer->scene);

	auto &&[state_ret, adv, rot] = state_machine_navigator(data, state_global, corners, lines);

	state_global = state_ret;
	set_speeds(0, adv, rot);

	// draw robot in viewer
	robot_room_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
	const double angle = qRadiansToDegrees(std::atan2(robot_pose.rotation()(1, 0), robot_pose.rotation()(0, 0)));
	robot_room_draw->setRotation(angle);



   // update GUI
   time_series_plotter->update();
   lcdNumber_x->display(robot_pose.translation().x());
   lcdNumber_y->display(robot_pose.translation().y());
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SpecificWorker::localise(RoboCompLidar3D::TPoints filter_data)
{
	const auto &[m_corners, lines] = room_detector.compute_corners(filter_data, &viewer->scene);
	Corners m_room_corners = nominal_rooms[habitacion].transform_corners_to(robot_pose.inverse());

	Match match = hungarian.match(m_corners, m_room_corners, 2000);

	float max_match_error = 99999.f;

	if (!match.empty())
	{
       const auto max_error_iter = std::ranges::max_element(match, [](const auto &a, const auto &b)
           { return std::get<2>(a) < std::get<2>(b); });
       max_match_error = static_cast<float>(std::get<2>(*max_error_iter));
       time_series_plotter->addDataPoint(0,max_match_error);
	}

	Eigen::MatrixXd W(m_corners.size() * 2, 3);
	Eigen::VectorXd b(m_corners.size() * 2);

	for (auto &&[i, m]: match | iter::enumerate)
	{
		auto &[meas_c, nom_c, _] = m;
		auto &[p_meas, __, ___] = meas_c;
		auto &[p_nom, ____, _____] = nom_c;
		b(2 * i)     = p_nom.x() - p_meas.x();
		b(2 * i + 1) = p_nom.y() - p_meas.y();
		W.block<1, 3>(2 * i, 0)     << 1.0, 0.0, -p_meas.y();
		W.block<1, 3>(2 * i + 1, 0) << 0.0, 1.0, p_meas.x();
	}
	// estimate new pose with pseudoinverse
	const Eigen::Vector3d r = (W.transpose() * W).inverse() * W.transpose() * b;
	std::cout << r << std::endl;
	qInfo() << "--------------------";


	if (r.array().isNaN().any())
		return;


	robot_pose.translate(Eigen::Vector2d(r(0), r(1)));
	robot_pose.rotate(r[2]);

	robot_room_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
	double angle = std::atan2(robot_pose.rotation()(1, 0), robot_pose.rotation()(0, 0));
	robot_room_draw->setRotation(angle * 180 / M_PI);
	lcdNumber_angle->display(angle);
}

void SpecificWorker::set_speeds(float vert, float adv, float rot)
{
	try {
		omnirobot_proxy->setSpeedBase(vert, adv, rot);
		lcdNumber_adv->display(adv);
		lcdNumber_rot->display(rot);

	}
	catch(const Ice::Exception &e) {
		std::cout << e << " Conexión con Laser\n";
	}
}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::state_machine(RoboCompLidar3D::TPoints filter_data, State state)
{
	switch (state)
	{
	case State::IDLE:
		qInfo() << "IDLE";
		return fwd(filter_data);
		break;
	case State::FORWARD:
		qInfo() << "FORWARD";
		return fwd(filter_data);
		break;
	case State::TURN:
		qInfo() << "TURN";
		return turn(filter_data);
		break;
	case State::FOLLOW_WALL:
		qInfo() << "FOLLOW_WALL";
		return wall(filter_data);
		break;
	case State::SPIRAL:
		qInfo() << "SPIRAL";
		return spiral(filter_data);
		break;

	}
}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::state_machine_navigator(RoboCompLidar3D::TPoints filter_data, State state, Corners corners, Lines lines)
{
	qInfo() << to_string(state);

	switch (state)
	{
	case State::GOTO_ROOM_CENTER:
		return goto_room_center(filter_data);
		break;
	case State::TURN:
		return turn_to_color(filter_data);
		break;
	case State::GOTO_DOOR:
		return goto_door(filter_data);
		break;
	case State::ORIENT_TO_DOOR:
		return orient_to_door(filter_data);
		break;
	}
}

SpecificWorker::RetVal SpecificWorker::goto_room_center(const RoboCompLidar3D::TPoints &points)
{
	auto centro = center_estimator.estimate(points);

	if (!centro)
		return {State::GOTO_ROOM_CENTER, 1.0, 0.0};

	// static QGraphicsEllipseItem *item = nullptr;
	// if (item != nullptr) delete item;
	// item = viewer->scene.addEllipse(-100, 100, 200, 200, QPen(Qt::red, 3), QBrush(Qt::red, Qt::SolidPattern));
	// item->setPos(centro->x(), centro->y());

	float k = 0.5f;
	auto angulo = atan2(centro->x(), centro->y());

	float dist = centro.value().norm();
	if (dist < 300) return {State::TURN, 0.0, 0.0};

	float vrot = k * angulo;
	float brake = exp(-angulo * angulo / (M_PI/10));
	qInfo() << "Brake: " << brake;
	float adv = 1000.0 * brake;

	return {State::GOTO_ROOM_CENTER, adv, vrot};
}

SpecificWorker::RetVal SpecificWorker::turn_to_color(RoboCompLidar3D::TPoints& puntos)
{
	auto const &[success, spin] = image_processor.check_colour_patch_in_image(this->camera360rgb_proxy, color_act);

	qInfo() << " Es red: " << success;

	if (success)
	{
		localised = true;
		return {State::GOTO_DOOR, 0.0, 0.0};
	}
	return {State::TURN, 0.0, 0.3 * spin};
}

SpecificWorker::RetVal SpecificWorker::goto_door(const RoboCompLidar3D::TPoints& puntos)
{
	if (door_detector.doors().empty())
		return {State::GOTO_DOOR, 0.0, 0.0};

	Doors doors = door_detector.doors();
	Door door = doors.front();

	auto centro = door.center_before(Eigen::Vector2d(robot_pose.translation().x(), robot_pose.translation().y()));

	float k = 0.5f;
	auto angulo = atan2(centro.x(), centro.y());

	float dist = centro.norm();
	if (dist < 400) return {State::ORIENT_TO_DOOR, 0.0, 0.0};

	float vrot = k * angulo;
	float brake = exp(-angulo * angulo / (M_PI/10));
	qInfo() << "Brake: " << brake;
	float adv = 1000.0 * brake;

	return {State::GOTO_DOOR, adv, vrot};
}

SpecificWorker::RetVal SpecificWorker::orient_to_door (const RoboCompLidar3D::TPoints& puntos)
{
	if (door_detector.doors().empty())
		return {State::ORIENT_TO_DOOR, 0.0, 0.0};

	Doors doors = door_detector.doors();
	Door door = doors.front();

	auto centro = door.center();

	float k = 0.5f;
	auto angulo = atan2(centro.x(), centro.y());

	if (angulo < 0.01)
	{
		auto start_time = std::chrono::steady_clock::now();

		// 2. Define the duration: 2 seconds.
		auto duration = std::chrono::seconds(2);

		// 3. Define the end time.
		auto end_time = start_time + duration;

		// --- Loop Section ---
		while (std::chrono::steady_clock::now() < end_time) {
			// This is the code you want to repeat:
			set_speeds(0, 1000.0, 0.0);

			// OPTIONAL: Add a small delay (sleep) to prevent the loop from running
			// too fast and consuming too much CPU. The duration of this delay
			// depends on how quickly your device can accept new speed commands.
			// For example, 10 milliseconds:
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		if (habitacion == 0)
		{
			habitacion = 1;
			color_act = "green";
		}
		else
		{
			habitacion = 0;
			color_act = "red";
		}

		localised = false;

		return {State::GOTO_ROOM_CENTER, 1000.0, 0.0};
	}
	//
	float vrot = k * angulo;
	// float brake = exp(-angulo * angulo / M_PI/3);
	// float adv = 1000.0 * brake;

	return {State::ORIENT_TO_DOOR, 0.0, vrot};
}


std::tuple<SpecificWorker::State, float, float> SpecificWorker::fwd(RoboCompLidar3D::TPoints puntos)
{
	auto pC = filter_ahead(puntos, 0);
	if(pC.empty())
		return {State::TURN, 0.0, 1.0};
	auto min_C = std::min_element(pC.begin(), pC.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });
	auto min = std::min_element(puntos.begin(), puntos.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });

	if (min_C->r<550)
	{
		derecha = min_C->phi >= 0 || (min->r < 550 && min->phi >= 0);

		return {State::TURN, 0.0, 0.0};
	}


	return{State::FORWARD, 1000.0, 0.0};
}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::turn(RoboCompLidar3D::TPoints puntos) //NO GIRA IZQ
{
	auto pC = filter_ahead(puntos, 0);
	if(pC.empty())
		return {State::TURN, 0.0, 1.0};
	auto min_C = std::min_element(pC.begin(), pC.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });

	if (min_C->r<550)
	{
		if (derecha)
			return{State::TURN, 0.0, -1.0};
		return {State::TURN, 0.0, 1.0};
	}

	srand(time(NULL));
	int rand_num = rand() % 3;

	switch (rand_num)
	{
	case 0:
		return {State::SPIRAL, 0.0, 0.0};
	case 1:
		return {State::FORWARD, 0.0, 0.0};
	case 2:
		return {State::FOLLOW_WALL, 0.0, 0.0};

	}

}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::wall(RoboCompLidar3D::TPoints puntos)
{
	auto pC = filter_ahead(puntos, 0);
	auto min_C = std::min_element(pC.begin(), pC.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });
	auto min = std::min_element(puntos.begin(), puntos.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });

	if (min->r > 550 || min_C->r > 650)
	{
			return {State::FORWARD, 0.0, 0.0};
	}

	if (derecha)
		return {State::FOLLOW_WALL, 0.0, -0.5};
	return {State::FOLLOW_WALL, 0.0, 0.5};
}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::spiral(RoboCompLidar3D::TPoints puntos)
{
	auto pC = filter_ahead(puntos, 0);
	if(pC.empty())
		return {State::TURN, 0.0, 1.0};

	auto min_C = std::min_element(pC.begin(), pC.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });

	auto min = std::min_element(puntos.begin(), puntos.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });

	int sign = derecha ? -1 : 1;
	qInfo() << "Velocidad rotacion: " << spir_rot;

	if (min_C->r < 550)
	{
		derecha = min_C->phi >= 0 || (min->r < 550 && min->phi >= 0);
		spir_rot = 1.0;
		return {State::TURN, 0.0, 0.0};
	}

	if (spir_rot < 0.005)
	{
		spir_rot = 1.0;
		derecha = !derecha;
		return {State::SPIRAL, 0.0, 0.0};
	}
	spir_rot -= 0.007;

	return {State::SPIRAL, spir_speed, spir_rot * sign};
}


RoboCompLidar3D::TPoints SpecificWorker::filter_ahead(RoboCompLidar3D::TPoints points,int n)
{
	RoboCompLidar3D::TPoints puntos;
	float inicio,fin=0;

	switch(n)
	{
	case 0:
		inicio=std::numbers::pi/4; //IZQUIERDA
		fin=-std::numbers::pi/4; //DERECHA
		break;
	case 1:
		inicio=-std::numbers::pi/4;
		fin=-std::numbers::pi/2;
		break;
	case 2:
		inicio=std::numbers::pi/2;
		fin=std::numbers::pi/4;
		break;
	default:
		inicio=std::numbers::pi/2;
		fin=-std::numbers::pi/2;
	}
	for (auto i = 0; i < points.size(); ++i)
	{
		if (points[i].phi < inicio && points[i].phi > fin)
		{
			puntos.push_back(points[i]);
		}
	}
	return puntos;
}

RoboCompLidar3D::TPoints SpecificWorker::filter_min_distance(RoboCompLidar3D::TPoints points)
{
	RoboCompLidar3D::TPoints salida;
	// Agrupar por phi y obtener el mínimo de r por grupo en una línea, usando push_back para almacenar en el vector
	for (auto&& group : iter::groupby(points, [](const auto& p)
	{
		float factor = std::pow(10, 2);  // Potencia de 10 para mover el punto decimal
		return std::round(p.phi * factor) / factor;  // Redondear y devolver con la cantidad deseada de decimales
	})) {
		auto min_r = std::min_element(group.second.begin(), group.second.end(),
			[](const auto& p1, const auto& p2) { return p1.r < p2.r; });
		salida.emplace_back(*min_r);
	}

	return salida;
}
void SpecificWorker::draw_lidar(const auto &points, QGraphicsScene* scene)
{
	static std::vector<QGraphicsItem*> draw_points;
	for (const auto &p : draw_points)
	{
		scene->removeItem(p);
		delete p;
	}
	draw_points.clear();

	const QColor color("LightGreen");
	const QPen pen(color, 10);
	//const QBrush brush(color, Qt::SolidPattern);
	for (const auto &p : points)
	{
		const auto dp = scene->addRect(-25, -25, 50, 50, pen);
		dp->setPos(p.x, p.y);
		draw_points.push_back(dp);   // add to the list of points to be deleted next time
	}
}

void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}



//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}

//UBSCRIPTION to sendData method from JoystickAdapter interface
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
	//subscribesToCODE

}

/**************************************/
// From the RoboCompCamera360RGB you can call this methods:
// RoboCompCamera360RGB::TImage this->camera360rgb_proxy->getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)

/**************************************/
// From the RoboCompCamera360RGB you can use this types:
// RoboCompCamera360RGB::TRoi
// RoboCompCamera360RGB::TImage

/**************************************/


/**************************************/
// From the RoboCompDifferentialRobot you can call this methods:
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->correctOdometer(int x, int z, float alpha)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->getBasePose(int x, int z, float alpha)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->getBaseState(RoboCompGenericBase::TBaseState state)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->resetOdometer()
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->setOdometer(RoboCompGenericBase::TBaseState state)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->setOdometerPose(int x, int z, float alpha)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->setSpeedBase(float adv, float rot)
// RoboCompDifferentialRobot::void this->differentialrobot_proxy->stopBase()

/**************************************/
// From the RoboCompDifferentialRobot you can use this types:
// RoboCompDifferentialRobot::TMechParams

/**************************************/
// From the RoboCompLaser you can call this methods:
// RoboCompLaser::TLaserData this->laser_proxy->getLaserAndBStateData(RoboCompGenericBase::TBaseState bState)
// RoboCompLaser::LaserConfData this->laser_proxy->getLaserConfData()
// RoboCompLaser::TLaserData this->laser_proxy->getLaserData()

/**************************************/
// From the RoboCompLaser you can use this types:
// RoboCompLaser::LaserConfData
// RoboCompLaser::TData

