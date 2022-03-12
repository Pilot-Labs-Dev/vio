// This file is part of VIO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// VIO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// VIO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <vio/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <vio/global.h>
#include <vio/global_optimizer.h>
#include <vio/frame.h>
#include <vio/point.h>
#include <vio/feature.h>
#include <vio/config.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#if VIO_DEBUG
#include <sys/types.h>
#include <sys/stat.h>
#endif
namespace vio {


    BA_Glob::BA_Glob(Map& map) :map_(map),thread_(NULL)
    {
        optimizer_=std::make_unique<g2o::SparseOptimizer>();
        optimizer_->setVerbose(false);
        /*g2o::BlockSolver_6_3::LinearSolverType * linearSolver=new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();*/
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver=g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
        std::unique_ptr<g2o::OptimizationAlgorithm> solver(new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))));
        optimizer_->setAlgorithm(std::move(solver));
        // setup camera
        cam_params_ =std::make_shared<g2o::CameraParameters>(1.0, Vector2d(0.,0.), 0.);
        cam_params_->setId(0);
        if (!optimizer_->addParameter(cam_params_)) {
            assert(false && "Camera initialization in BA");
        }
#if VIO_DEBUG
        log_ =fopen((std::string(PROJECT_DIR)+"/loop_closure_log.txt").c_str(),"w+");
        assert(log_);
        chmod((std::string(PROJECT_DIR)+"/loop_closure_log.txt").c_str(), ACCESSPERMS);
#endif
    }

    BA_Glob::~BA_Glob()
    {
        stopThread();
    }

    void BA_Glob::startThread()
    {
        thread_ = new boost::thread(&BA_Glob::updateLoop, this);
    }

    void BA_Glob::stopThread()
    {

        if(thread_ != NULL)
        {
            thread_->interrupt();
            usleep(5000);
            thread_->join();
            thread_ = NULL;
        }
#if VIO_DEBUG
        fclose(log_);
#endif
    }
    void BA_Glob::updateLoop()
    {
        while(!boost::this_thread::interruption_requested())
        {
            boost::unique_lock< boost::mutex > lk( mtx_);
            while(map_.keyframes_.empty() || new_keyframe_ == false)
                cond_.wait(lk);
#if VIO_DEBUG
            fprintf(log_,"[%s] BA loop run \n",
                    vio::time_in_HH_MM_SS_MMM().c_str());
#endif
            new_keyframe_=false;
            // init g2o
            g2o::OptimizableGraph::VertexContainer points;
            ba_mux_.lock();
            // Go through all Keyframes
            v_id_ = 0;
            auto end=map_.keyframes_.end();
            auto end_1=end--;
            for(auto it_kf=map_.keyframes_.begin();it_kf!=map_.keyframes_.end();++it_kf)
            {
                // New Keyframe Vertex
                if(it_kf !=map_.keyframes_.end() && it_kf !=end && it_kf !=end_1){
                    (*it_kf)->v_kf_ = createG2oFrameSE3(*it_kf,true);
                }else{
                    (*it_kf)->v_kf_ = createG2oFrameSE3(*it_kf,false);
                }
                optimizer_->addVertex((*it_kf)->v_kf_);
                for(auto& it_ftr:(*it_kf)->fts_)
                {
                    if(it_ftr->point==NULL)continue;
                    if(it_ftr->point->type_ != vio::Point::TYPE_GOOD)continue;
                    // for each keyframe add edges to all observed mapoints
                    if(it_ftr->point->pos_.hasNaN())continue;
                    if(it_ftr->point->pos_.norm()==0.)continue;
                    if(it_ftr->point->v_pt_ == NULL)
                    {
                        // mappoint-vertex doesn't exist yet. create a new one:
                        it_ftr->point->v_pt_ = createG2oPoint(it_ftr->point->pos_);
                        optimizer_->addVertex(it_ftr->point->v_pt_);
                    }
                    optimizer_->addEdge(createG2oEdgeSE3((*it_kf)->v_kf_, it_ftr->point->v_pt_, vk::project2d(it_ftr->f),
                                                                         true,
                                                                         Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()*Config::lobaRobustHuberWidth()));
                }
            }
            for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer_->vertices().begin();
                 it != optimizer_->vertices().end(); ++it) {
                auto v = std::static_pointer_cast<g2o::OptimizableGraph::Vertex>(it->second);
                if (v->dimension() == 3 && v->edges().size()>2) points.push_back(v);
            }
            // Optimization
            if(points.empty()){
                optimizer_->clear();
                ba_mux_.unlock();
                continue;
            }
            optimizer_->initializeOptimization();
            optimizer_->computeActiveErrors();
            g2o::StructureOnlySolver<3> structure_only_ba;
            structure_only_ba.calc(points, vio::Config::lobaNumIter());

#if VIO_DEBUG
            fprintf(log_,"[%s] init error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer_->activeChi2());
#endif
            if(optimizer_->optimize(vio::Config::lobaNumIter())<1){
                optimizer_->clear();
                ba_mux_.unlock();
                continue;
            }
#if VIO_DEBUG
            fprintf(log_,"[%s] end error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer_->activeChi2());
#endif
            // Update Keyframe and MapPoint Positions
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end();++it_kf)
            {
                (*it_kf)->T_f_w_ = SE2_5(SE3((*it_kf)->v_kf_->estimate().rotation().toRotationMatrix(),
                                        (*it_kf)->v_kf_->estimate().translation()));
                for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
                {
                    if((*it_ftr)->point == NULL)
                        continue;
                    if((*it_ftr)->point->v_pt_ == NULL)
                        continue;       // mp was updated before
                    (*it_ftr)->point->pos_ = (*it_ftr)->point->v_pt_->estimate();
                    (*it_ftr)->point->v_pt_.reset();
                }
            }
            optimizer_->clear();
            ba_mux_.unlock();
        }
    }

   std::shared_ptr<g2o::VertexSE3Expmap>
   BA_Glob::createG2oFrameSE3(FramePtr frame, bool state)
   {
       std::shared_ptr<g2o::VertexSE3Expmap> v= std::make_shared<g2o::VertexSE3Expmap>();
       ++v_id_;
       v->setId(v_id_);
       // not all frames are fixed
       v->setFixed(state);
       v->setEstimate(g2o::SE3Quat(frame->se3().unit_quaternion(), frame->se3().translation()));
       return v;
   }

   std::shared_ptr<g2o::VertexPointXYZ>
   BA_Glob::createG2oPoint(Vector3d pos)
   {
       ++v_id_;
       std::shared_ptr<g2o::VertexPointXYZ> v =std::make_shared<g2o::VertexPointXYZ>();
       v->setId(v_id_);
       //v->setFixed(false);
       v->setMarginalized(true);
       v->setEstimate(pos);
       return v;

   }

    std::shared_ptr<g2o::EdgeProjectXYZ2UV> BA_Glob::createG2oEdgeSE3(
            std::shared_ptr<g2o::VertexSE3Expmap> v_frame,
            std::shared_ptr<g2o::VertexPointXYZ> v_point,
                     const Vector2d& f_up,
                     bool robust_kernel,
                     double huber_width,
                     double weight)
   {
       std::shared_ptr<g2o::EdgeProjectXYZ2UV> e= std::make_shared<g2o::EdgeProjectXYZ2UV>();
       e->vertices()[0]=v_point;
       e->vertices()[1]=v_frame;
       e->setMeasurement(f_up);
       e->information() = weight*Eigen::Matrix2d::Identity();
       g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
       rk->setDelta(huber_width);
       e->setRobustKernel(rk);
       e->setParameterId(0, 0); //old: e->setId(v_point->id());
       return e;
   }
   void BA_Glob::reset_map(){
        for(auto&& f:map_.keyframes_){
            f->v_kf_.reset();
            for(auto p:f->fts_)
                if(p->point!=NULL)
                    p->point->v_pt_.reset();
        }
    }

} // namespace vio
