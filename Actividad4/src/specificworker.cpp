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
		room_draw = viewer_room->scene.addRect(nominal_rooms[habitacion].rect(), QPen(Qt::black, 30));


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

	if(localised) localise(data);
	show_loc_error(data);

	auto centro = center_estimator.estimate(data);

	if (centro)
	{
		static QGraphicsEllipseItem *item = nullptr;
		if (item != nullptr) delete item;
		item = viewer->scene.addEllipse(-100, 100, 200, 200, QPen(Qt::red, 3), QBrush(Qt::red, Qt::SolidPattern));
		item->setPos(centro->x(), centro->y());
	}

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

	Match match = hungarian.match(m_corners, m_room_corners);

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

	if (r.array().isNaN().any())
		return;

	robot_pose.translate(Eigen::Vector2d(r(0), r(1)));
	robot_pose.rotate(r[2]);

	robot_room_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
	double angle = std::atan2(robot_pose.rotation()(1, 0), robot_pose.rotation()(0, 0));
	robot_room_draw->setRotation(angle * 180 / M_PI);
	lcdNumber_angle->display(angle);
}

bool SpecificWorker::update_robot_pose(const Corners &corners, const Match &match)
{
	Eigen::MatrixXd W(corners.size() * 2, 3);
	Eigen::VectorXd b(corners.size() * 2);

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

	if (r.array().isNaN().any())
		return false;

	robot_pose.translate(Eigen::Vector2d(r(0), r(1)));
	robot_pose.rotate(r[2]);

	robot_room_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
	double angle = std::atan2(robot_pose.rotation()(1, 0), robot_pose.rotation()(0, 0));
	robot_room_draw->setRotation(angle * 180 / M_PI);
	lcdNumber_angle->display(angle);

	return true;
}

float SpecificWorker::show_loc_error(RoboCompLidar3D::TPoints filter_data)
{
	const auto &[m_corners, lines] = room_detector.compute_corners(filter_data, &viewer->scene);
	Corners m_room_corners = nominal_rooms[habitacion].transform_corners_to(robot_pose.inverse());

	Match match = hungarian.match(m_corners, nominal_rooms[habitacion].transform_corners_to(robot_pose.inverse()));

	float max_match_error = 99999.f;

	if (!match.empty())
	{
		const auto max_error_iter = std::ranges::max_element(match, [](const auto &a, const auto &b)
			{ return std::get<2>(a) < std::get<2>(b); });
		max_match_error = static_cast<float>(std::get<2>(*max_error_iter));
		time_series_plotter->addDataPoint(0,max_match_error);
	}

	return max_match_error;
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
	qInfo() << to_string(state);
	label_state_name->setText(to_string(state));
	
	switch (state)
	{
	case State::IDLE:
		return fwd(filter_data);
		break;
	case State::FORWARD:
		return fwd(filter_data);
		break;
	case State::TURN:
		return turn(filter_data);
		break;
	case State::FOLLOW_WALL:
		return wall(filter_data);
		break;
	case State::SPIRAL:
		return spiral(filter_data);
		break;

	}
}

std::tuple<SpecificWorker::State, float, float> SpecificWorker::state_machine_navigator(RoboCompLidar3D::TPoints filter_data, State state, Corners corners, Lines lines)
{
	qInfo() << to_string(state);
	label_state_name->setText(to_string(state));

	switch (state)
	{
	case State::GOTO_ROOM_CENTER:
		return goto_room_center(filter_data);
		break;
	case State::TURN:
		return turn_to_color(corners);
		break;
	case State::GOTO_DOOR:
		return goto_door(filter_data);
		break;
	case State::ORIENT_TO_DOOR:
		return orient_to_door(filter_data);
		break;
	case State::CROSS_DOOR:
		return cross_door(filter_data);
		break;
	}
}

SpecificWorker::RetVal SpecificWorker::goto_room_center(const RoboCompLidar3D::TPoints &points)
{
	auto centro = center_estimator.estimate(points);

	if (!centro)
		return {State::GOTO_ROOM_CENTER, 1.0, 0.0};

	float k = 1.0f;
	auto angulo = atan2(centro->x(), centro->y());

	float dist = centro.value().norm();
	if (dist < 100) return {State::TURN, 0.0, 0.0};

	float vrot = k * angulo;
	float brake = exp(-angulo * angulo / (M_PI/10));
	float adv = 1000.0 * brake;

	return {State::GOTO_ROOM_CENTER, adv, vrot};
}

SpecificWorker::RetVal SpecificWorker::turn_to_color(const Corners &corners)
{
	static std::vector<QGraphicsItem*> g_items;
	for (auto &item : g_items)
	{
		viewer_room->scene.removeItem(item);
		delete item;
	}
	g_items.clear();

    const auto &[success, room_index, left_right] = image_processor.check_colour_patch_in_image(camera360rgb_proxy, this->label_img);
    if (success)
    {
        //habitacion = room_index;
        const auto m = hungarian.match(corners,nominal_rooms[habitacion].corners() );
        if (m.empty())
        {
            qInfo() << __FUNCTION__ << "empty match";
        };
        if (m.size() < 3)
        {
            qInfo() << __FUNCTION__ << "m size < 3";
            return{State::TURN, 0.0f, left_right*params.RELOCAL_ROT_SPEED};
        }
        const auto max_error_iter = std::ranges::max_element(m, [](const auto &a, const auto &b)
                                { return std::get<2>(a) < std::get<2>(b); });
        if (const auto max_match_error = std::get<2>(*max_error_iter); max_match_error > params.RELOCAL_DONE_MATCH_MAX_ERROR)
        {
            qInfo() << __FUNCTION__ << "match error > " << params.RELOCAL_DONE_MATCH_MAX_ERROR;
            return{State::TURN, 0.0f, left_right*params.RELOCAL_ROT_SPEED};
        }
        // update robot pose to have a fresh value
        update_robot_pose(corners, m);

        ///////////////////////////////////////////////////////////////////////

        // save doors to nominal_room
        auto doors = door_detector.doors();
        if (doors.empty()) { qWarning() << __FUNCTION__ << "empty doors"; return{State::TURN, 0.0f, left_right*params.RELOCAL_ROT_SPEED};}
        for (auto &d : doors)
        {
            d.global_p1 = nominal_rooms[habitacion].get_projection_of_point_on_closest_wall(robot_pose.cast<float>() * d.p1);
            d.global_p2 = nominal_rooms[habitacion].get_projection_of_point_on_closest_wall(robot_pose.cast<float>() * d.p2);
        }
        nominal_rooms[habitacion].doors = doors;
        // choose door to go

    	for (int i = 0; i < nominal_rooms[habitacion].doors.size(); i++)
    	{
    		if (i != door_crossing.leaving_door_index)
    		{
    			current_door = i;
    			break;
    		}
    	}
        // we need to match the current selected nominal door to the successive local doors detected during the approach
        // select the local door closest to the selected nominal door
        const auto dn = nominal_rooms[habitacion].doors[current_door];
        const auto ds = door_detector.doors();
        const auto sd = std::ranges::min_element(ds, [dn, this](const auto &a, const auto &b)
                {  return (a.center() - robot_pose.inverse().cast<float>() * dn.center_global()).norm() <
                          (b.center() - robot_pose.inverse().cast<float>() * dn.center_global()).norm(); });
        // sd is the closest local door to the selected nominal door. Update nominal door with local values
        nominal_rooms[habitacion].doors[current_door].p1 = sd->p1;
        nominal_rooms[habitacion].doors[current_door].p2 = sd->p2;
        localised = true;

    	for (auto door : door_detector.doors())
    	{
    		door.global_p1 = nominal_rooms[habitacion].get_projection_of_point_on_closest_wall(robot_pose.cast<float>() * door.p1.cast<float>());
    		door.global_p2 = nominal_rooms[habitacion].get_projection_of_point_on_closest_wall(robot_pose.cast<float>() * door.p2.cast<float>());
    		auto a = viewer_room->scene.addEllipse(-100, -100, 200, 200, QPen(QColor("blue")), QBrush(QColor("blue")));
    		auto b = viewer_room->scene.addEllipse(-100, -100, 200, 200, QPen(QColor("blue")), QBrush(QColor("blue")));
    		a->setPos(door.global_p1.x(), door.global_p1.y());
    		b->setPos(door.global_p2.x(), door.global_p2.y());

    		const auto door_draw = viewer_room->scene.addLine(door.global_p1.x(), door.global_p1.y(), door.global_p2.x(), door.global_p2.y(), QPen(Qt::cyan, 30));

    		g_items.push_back(a);
    		g_items.push_back(b);
    		g_items.push_back(door_draw);
    	}

    	nominal_rooms[habitacion].doors = door_detector.doors();
    	// srand(time(NULL));
    	// current_door = rand() % door_detector.doors().size();
        return {State::GOTO_DOOR, 0.0f, 0.0f};  // SUCCESS
    }
    // continue turning
    return {State::TURN, 0.0f, left_right*params.RELOCAL_ROT_SPEED};
}

SpecificWorker::RetVal SpecificWorker::goto_door(const RoboCompLidar3D::TPoints &points)
{
    Doors doors;
    // Exit conditions
    if ( doors = door_detector.doors(); doors.empty())
    {
        qInfo() << __FUNCTION__ << "No doors detected";
        return {State::GOTO_DOOR, 0.f, 0.f};
    }
    // select from doors, the one closest to the nominal door
    Door *target_door = &doors[current_door];

    qInfo() << target_door->p1.x() << target_door->p1.y();


    // distance to target is less than threshold, stop and switch to ORIENT_TO_DOOR
    constexpr float offset = 600.f;
    const auto target = target_door->center_before(robot_pose.translation(), offset);
    const auto dist_to_door = target.norm();

	auto centro = target_door->center_before(Eigen::Vector2d(robot_pose.translation().x(), robot_pose.translation().y()));

	float k = 1.0f;
	auto angulo = atan2(centro.x(), centro.y());

	float dist = centro.norm();
	if (dist < 600) return {State::ORIENT_TO_DOOR, 0.0, 0.0};

	float vrot = k * angulo;
	float brake = exp(-angulo * angulo / (M_PI/10));
	float adv = 1000.0 * brake;

	return {State::GOTO_DOOR, adv, vrot};
}

SpecificWorker::RetVal SpecificWorker::orient_to_door(const RoboCompLidar3D::TPoints &points)
{
	const auto doors = door_detector.doors();

	const auto sd = std::ranges::min_element(doors, [](const auto &a, const auto &b)
		   {  return std::fabs(a.center_angle()) < std::fabs(b.center_angle());} );

	auto centro = sd->center();

	float k = 0.5f;
	auto angulo = atan2(centro.x(), centro.y());

	if (abs(angulo) < 0.01)
	{
		localised = false;
		return {State::CROSS_DOOR, 0.5, 0.0};
	}

	float vrot = k * angulo;

	return {State::ORIENT_TO_DOOR, 0.0, vrot};
}

SpecificWorker::RetVal SpecificWorker::cross_door(const RoboCompLidar3D::TPoints &points)
{
	static bool first_time = true;
	static std::chrono::time_point<std::chrono::system_clock> start;

	// Exit condition: the robot has advanced 1000 or equivalently 2 seconds at 500 mm/s
	if (first_time)
	{
		first_time = false;
		start = std::chrono::high_resolution_clock::now();
		return {State::CROSS_DOOR, 1000.0f, 0.0f};
	}
	else
	{
		const auto elapsed = std::chrono::high_resolution_clock::now() - start;
		//qInfo() << __FUNCTION__ << "Elapsed time crossing door: "
		//         << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms";
		if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > 3000)
		{
			first_time = true;
			const auto &leaving_door = nominal_rooms[habitacion].doors[current_door];
			// // Update indices to the new room
			// int next_door_idx = leaving_door.connects_to_door;
			//habitacion = next_room_idx;
			habitacion = !habitacion;
			viewer_room->scene.removeItem(room_draw);
			delete room_draw;
			room_draw = viewer_room->scene.addRect(nominal_rooms[habitacion].rect(), QPen(Qt::black, 30));
			lcdNumber_room->display(habitacion);
			//current_door = next_door_idx;


			// Compute robot pose based on the door in the new room frame.
			door_detector.detect(points);

			nominal_rooms[habitacion].doors = door_detector.doors();
			if (!nominal_rooms[habitacion].doors.empty())
			{
				const auto &entering_door = nominal_rooms[habitacion].doors[current_door]; // door we are entering now
				Eigen::Vector2f door_center = entering_door.center_global(); //
				// Vector from door to origin (0,0) is -door_center
				const float angle = std::atan2(-door_center.x(), -door_center.y());
				// robot_pose now must be translated so it is drawn in the new room correctly
				robot_pose.setIdentity();
				door_center.y() -= 500; // place robot 500 mm inside the room
				robot_pose.translate(door_center.cast<double>());
				robot_pose.rotate(0);
				//qInfo() << __FUNCTION__ << "Robot localized in NEW room " << habitacion << " at door " << current_door;
				std::cout << door_center.x() << " " << door_center.y() << " " << angle << std::endl;
			}

			localised = true;
			// Continue navigation in the new room
			door_crossing.track_entering_door(door_detector.doors());
			// door_crossing.set_entering_data(door_crossing.leaving_room_index, nominal_rooms);

			return {State::GOTO_ROOM_CENTER, 0.f, 0.f};

		}
		else // keep crossing
			return {State::CROSS_DOOR, 1000.f, 0.f};
	}
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
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data){}

