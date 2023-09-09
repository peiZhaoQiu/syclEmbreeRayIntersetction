#include <sycl/sycl.hpp>
#include <embree4/rtcore.h> // Include the appropriate Embree header
#include <limits>
#include <stdio.h>
#include <math.h>
#include "Ray.hpp"

/*
 * We will register this error handler with the device in initializeDevice(),
 * so that we are automatically informed on errors.
 * This is extremely helpful for finding bugs in your code, prevents you
 * from having to add explicit error checking to each Embree API call.
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
  printf("error %d: %s\n", error, str);
}

/*
 * Embree has a notion of devices, which are entities that can run 
 * raytracing kernels.
 * We initialize our device here, and then register the error handler so that 
 * we don't miss any errors.
 *
 * rtcNewDevice() takes a configuration string as an argument. See the API docs
 * for more information.
 *
 * Note that RTCDevice is reference-counted.
 */
RTCDevice initializeDevice()
{
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  return device;
}

/*
 * Create a scene, which is a collection of geometry objects. Scenes are 
 * what the intersect / occluded functions work on. You can think of a 
 * scene as an acceleration structure, e.g. a bounding-volume hierarchy.
 *
 * Scenes, like devices, are reference-counted.
 */
RTCScene initializeScene(RTCDevice device)
{
  RTCScene scene = rtcNewScene(device);

  /* 
   * Create a triangle mesh geometry, and initialize a single triangle.
   * You can look up geometry types in the API documentation to
   * find out which type expects which buffers.
   *
   * We create buffers directly on the device, but you can also use
   * shared buffers. For shared buffers, special care must be taken
   * to ensure proper alignment and padding. This is described in
   * more detail in the API documentation.
   */
  RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
  float* vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                     RTC_BUFFER_TYPE_VERTEX,
                                                     0,
                                                     RTC_FORMAT_FLOAT3,
                                                     3*sizeof(float),
                                                     3);

  unsigned* indices = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                          RTC_BUFFER_TYPE_INDEX,
                                                          0,
                                                          RTC_FORMAT_UINT3,
                                                          3*sizeof(unsigned),
                                                          1);

  if (vertices && indices)
  {
    vertices[0] = 0.f; vertices[1] = 0.f; vertices[2] = 0.f;
    vertices[3] = 1.f; vertices[4] = 0.f; vertices[5] = 0.f;
    vertices[6] = 0.f; vertices[7] = 1.f; vertices[8] = 0.f;

    indices[0] = 0; indices[1] = 1; indices[2] = 2;
  }

  /*
   * You must commit geometry objects when you are done setting them up,
   * or you will not get any intersections.
   */
  rtcCommitGeometry(geom);

  /*
   * In rtcAttachGeometry(...), the scene takes ownership of the geom
   * by increasing its reference count. This means that we don't have
   * to hold on to the geom handle, and may release it. The geom object
   * will be released automatically when the scene is destroyed.
   *
   * rtcAttachGeometry() returns a geometry ID. We could use this to
   * identify intersected objects later on.
   */
  rtcAttachGeometry(scene, geom);
  rtcReleaseGeometry(geom);

  /*
   * Like geometry objects, scenes must be committed. This lets
   * Embree know that it may start building an acceleration structure.
   */
  rtcCommitScene(scene);

  return scene;
}


/*
 * Cast a single ray with origin (ox, oy, oz) and direction
 * (dx, dy, dz).
 */
void castRay(RTCScene scene, 
             float ox, float oy, float oz,
             float dx, float dy, float dz)
{
  /*
   * The ray hit structure holds both the ray and the hit.
   * The user must initialize it properly -- see API documentation
   * for rtcIntersect1() for details.
   */
  struct RTCRayHit rayhit;
  rayhit.ray.org_x = ox;
  rayhit.ray.org_y = oy;
  rayhit.ray.org_z = oz;
  rayhit.ray.dir_x = dx;
  rayhit.ray.dir_y = dy;
  rayhit.ray.dir_z = dz;
  rayhit.ray.tnear = 0;
  rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rayhit.ray.mask = -1;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  /*
   * There are multiple variants of rtcIntersect. This one
   * intersects a single ray with the scene.
   */
  rtcIntersect1(scene, &rayhit);

  printf("%f, %f, %f: ", ox, oy, oz);
  if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    /* Note how geomID and primID identify the geometry we just hit.
     * We could use them here to interpolate geometry information,
     * compute shading, etc.
     * Since there is only a single triangle in this scene, we will
     * get geomID=0 / primID=0 for all hits.
     * There is also instID, used for instancing. See
     * the instancing tutorials for more information */
    printf("Found intersection on geometry %d, primitive %d at tfar=%f\n", 
           rayhit.hit.geomID,
           rayhit.hit.primID,
           rayhit.ray.tfar);
  }
  else
    printf("Did not find any intersection.\n");
}


// Define your Embree scene data here

//namespace sycl = cl::sycl;

// Define your SYCL kernel function
class RayIntersectionKernel {
public:
    RayIntersectionKernel(sycl::queue& q, RTCScene& scene, sycl::buffer<RTCRayHit>& rayResults, sycl::buffer<Ray>& rayBuffer, int raySize)
        : queue_(q), scene_(scene), rayResults_(rayResults), rayBuffer_(rayBuffer), raySize_(raySize) {}

    RTCRayHit castRay(Ray ray, RTCScene& scene) {
        struct RTCRayHit rayhit;
        rayhit.ray.org_x = ray.origin.x;
        rayhit.ray.org_y = ray.origin.y;
        rayhit.ray.org_z = ray.origin.z;
        rayhit.ray.dir_x = ray.direction.x;
        rayhit.ray.dir_y = ray.direction.y;
        rayhit.ray.dir_z = ray.direction.z;
        rayhit.ray.tnear = 0;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = ((unsigned int)-1);
        rtcIntersect1(scene, &rayhit);

        return rayhit;
    }

    void operator()() {
        RTCScene& scene = this->scene_;
        sycl::buffer<RTCRayHit>& rayResults = this->rayResults_;
        sycl::buffer<Ray>& rayBuffer = this->rayBuffer_;
        int raySize = this->raySize_;

        // Define and execute the kernel within the operator() function
        queue_.submit([&](sycl::handler& cgh) {
            auto rayResultsAcc = rayResults.template get_access<sycl::access::mode::write>(cgh);
            auto rayBufferAcc = rayBuffer.template get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class MyKernel>(
                sycl::range<1>(raySize),
                [=](sycl::id<1> idx) {
                    // Kernel code goes here
                    rayResultsAcc[idx] = castRay(rayBufferAcc[idx], scene);
                }
            );
        });
        queue_.wait();
    }

private:
    RTCScene& scene_;
    sycl::buffer<RTCRayHit>& rayResults_;
    sycl::buffer<Ray>& rayBuffer_;
    sycl::queue& queue_;
    int raySize_;
};

int main() {
    // Initialize Embree, create a scene, and set up rays here

    //sycl::queue queue(sycl::gpu_selector{}); // Choose your SYCL device selector
    sycl::queue queue(sycl::default_selector{});
    //sycl::device device = queue.get_device();
    sycl::context context = queue.get_context();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device);
    

    Ray ray1(Vec3f(0.33f,0.33f,-1.0f),Vec3f(0.0f,0.0f,1.0f));

    Ray ray2(Vec3f(1.0f,1.0f,-1.0f),Vec3f(0.0f,0.0f,1.0f));

    std::vector<Ray> rays;
    rays.push_back(ray1);
    rays.push_back(ray2);
    int raySize = rays.size();

    // Create a buffer to hold ray results (e.g., hit information)
    sycl::buffer<RTCRayHit, 1> rayResults(raySize);
    sycl::buffer<Ray,1> rayBuffer(rays.data(),sycl::range<1>(raySize));

    // Create a SYCL kernel functor and execute it

    RayIntersectionKernel kernel(queue, scene, rayResults, rayBuffer, raySize);


    auto resultAccessor =  rayResults.template get_access<sycl::access::mode::read>();

     for (int i = 0; i < raySize; ++i) {
        //std::cout << "Result[" << i << "] = " << resultAccessor[i] << std::endl;
        auto rayhit = resultAccessor[i];
        //std::cout << "Result[" << i << "] = " << resultAccessor[i].hit.geomID << std::endl;
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
        {
          /* Note how geomID and primID identify the geometry we just hit.
          * We could use them here to interpolate geometry information,
          * compute shading, etc.
          * Since there is only a single triangle in this scene, we will
          * get geomID=0 / primID=0 for all hits.
          * There is also instID, used for instancing. See
          * the instancing tutorials for more information */
          printf("Found intersection on geometry %d, primitive %d at tfar=%f\n", 
                rayhit.hit.geomID,
                rayhit.hit.primID,
                rayhit.ray.tfar);
        }
        else
          printf("Did not find any intersection.\n");

    }


    // Wait for the SYCL queue to finish
    //queue.wait();

    // Process the rayResults buffer with the intersection information

    // Cleanup Embree and SYCL resources

    return 0;
}