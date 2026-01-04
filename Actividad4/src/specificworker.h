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

/**
    \brief
    @author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
// #define HIBERNATION_ENABLED

#ifdef emit
#undef emit
#endif
#include <cppitertools/enumerate.hpp>
#include <cppitertools/groupby.hpp>
#include <cppitertools/range.hpp>
#include <expected>
#include <genericworker.h>
#include <time.h>
#include <tuple>
#include <vector>

#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include "door_crossing_tracker.h"
#include "door_detector.h"
#include "hungarian.h"
#include "image_processor.h"
#include "nominal_room.h"
#include "pointcloud_center_estimator.h"
#include "room_detector.h"
#include "time_series_plotter.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker final : public GenericWorker {
    Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int startup_check();

private:
    // params
    struct Params {
        float ROBOT_WIDTH = 460; // mm
        float ROBOT_LENGTH = 480; // mm
        float MAX_ADV_SPEED = 1000; // mm/s
        float MAX_ROT_SPEED = 1; // rad/s
        float MAX_SIDE_SPEED = 50; // mm/s
        float MAX_TRANSLATION = 500; // mm/s
        float MAX_ROTATION = 0.2;
        float STOP_THRESHOLD = 700; // mm
        float ADVANCE_THRESHOLD = ROBOT_WIDTH * 3; // mm
        float LIDAR_FRONT_SECTION = 0.2; // rads, aprox 12 degrees
        // wall
        float LIDAR_RIGHT_SIDE_SECTION = M_PI / 3; // rads, 90 degrees
        float LIDAR_LEFT_SIDE_SECTION = -M_PI / 3; // rads, 90 degrees
        float WALL_MIN_DISTANCE = ROBOT_WIDTH * 1.2;
        // match error correction
        float MATCH_ERROR_SIGMA = 150.f; // mm
        float DOOR_REACHED_DIST = 300.f;
        std::string LIDAR_NAME_LOW = "bpearl";
        std::string LIDAR_NAME_HIGH = "helios";
        QRectF GRID_MAX_DIM { -5000, 2500, 10000, -5000 };

        // relocalization
        float RELOCAL_CENTER_EPS = 300.f; // mm: stop when |mean| < eps
        float RELOCAL_KP = 0.002f; // gain to convert mean (mm) -> speed (magnitude)
        float RELOCAL_MAX_ADV = 300.f; // mm/s cap while re-centering
        float RELOCAL_MAX_SIDE = 300.f; // mm/s cap while re-centering
        float RELOCAL_ROT_SPEED = 0.3f; // rad/s while aligning
        float RELOCAL_DELTA = 5.0f * M_PI / 180.f; // small probe angle in radians
        float RELOCAL_MATCH_MAX_DIST = 2000.f; // mm for Hungarian gating
        float RELOCAL_DONE_COST = 500.f;
        float RELOCAL_DONE_MATCH_MAX_ERROR = 850.f;
    };
    Params params;
    const int ROBOT_LENGTH = 400;

    // state machine
    enum class State {
        IDLE,
        GOTO_DOOR,
        ORIENT_TO_DOOR,
        GOTO_ROOM_CENTER,
        TURN,
        CROSS_DOOR,
        FORWARD,
        FOLLOW_WALL,
        SPIRAL
    };

    inline const char* to_string(const State s) const
    {
        switch (s) {
        case State::IDLE:
            return "IDLE";
        case State::GOTO_DOOR:
            return "GOTO_DOOR";
        case State::TURN:
            return "TURN";
        case State::ORIENT_TO_DOOR:
            return "ORIENT_TO_DOOR";
        case State::GOTO_ROOM_CENTER:
            return "GOTO_ROOM_CENTER";
        case State::CROSS_DOOR:
            return "CROSS_DOOR";
        case State::FORWARD:
            return "FORWARD";
        case State::FOLLOW_WALL:
            return "FOLLOW_WALL";
        case State::SPIRAL:
            return "SPIRAL";

        default:
            return "UNKNOWN";
        }
    }
    State state_global = State::GOTO_ROOM_CENTER;

    using RetVal = std::tuple<State, float, float>;

    void localise(RoboCompLidar3D::TPoints filter_data);
    bool update_robot_pose(const Corners& corners, const Match& match);
    float show_loc_error(RoboCompLidar3D::TPoints filter_data);

    RetVal state_machine(RoboCompLidar3D::TPoints puntos, State state);
    RetVal state_machine_navigator(RoboCompLidar3D::TPoints filter_data, State state, Corners corners, Lines lines);
    RetVal turn_to_color(const Corners& corners);
    RetVal goto_room_center(const RoboCompLidar3D::TPoints& points);
    RetVal goto_door(const RoboCompLidar3D::TPoints& points);
    RetVal orient_to_door(const RoboCompLidar3D::TPoints& points);
    RetVal cross_door(const RoboCompLidar3D::TPoints& puntos);
    bool localised = false;
    bool cross_start = true;
    int habitacion = 0;
    int current_door = 0;

    RetVal fwd(RoboCompLidar3D::TPoints puntos);
    RetVal turn(RoboCompLidar3D::TPoints puntos);
    RetVal wall(RoboCompLidar3D::TPoints puntos);
    RetVal spiral(RoboCompLidar3D::TPoints puntos);
    bool fol_wall = false;
    bool derecha = false;
    float spir_rot = 0.6;
    float spir_speed = 1000.0;

    void set_speeds(float vert, float adv, float rot);

    // viewer and plotter
    AbstractGraphicViewer *viewer, *viewer_room;
    QGraphicsPolygonItem *robot_draw, *robot_room_draw;
    QGraphicsRectItem* room_draw;
    QRectF dimensions;
    QGraphicsPolygonItem* robot_polygon;
    std::unique_ptr<TimeSeriesPlotter> time_series_plotter;
    int match_error_graph; // To store the index of the speed graph
    void draw_lidar(const auto& points, QGraphicsScene* scene);

    // locations
    Eigen::Affine2d robot_pose;
    std::vector<NominalRoom> nominal_rooms { NominalRoom { 5500.f, 4000.f }, NominalRoom { 8000.f, 4000.f } };

    // tools
    rc::Room_Detector room_detector;
    rc::Hungarian hungarian;
    DoorCrossing door_crossing;
    DoorDetector door_detector;
    rc::ImageProcessor image_processor;

    // timing
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time = std::chrono::high_resolution_clock::now();
    rc::PointcloudCenterEstimator center_estimator;

    // filters
    RoboCompLidar3D::TPoints get_filtered_lidar_data();
    RoboCompLidar3D::TPoints filter_min_distance(RoboCompLidar3D::TPoints points);
    RoboCompLidar3D::TPoints filter_ahead(RoboCompLidar3D::TPoints points, int lado); // 0 adelante, 1 drcha, 2 izq

    bool startup_check_flag;

signals:
    // void customSignal();
};

#endif
