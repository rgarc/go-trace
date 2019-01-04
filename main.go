package main

import (
	"fmt"
  "math"
  "math/rand"
)

/**

  3D Vectors

**/

type Vec3 struct {
  e [3]float32
}

func (v Vec3) x() float32 {
  return v.e[0]
}

func (v Vec3) y() float32 {
  return v.e[1]
}

func (v Vec3) z() float32 {
  return v.e[2]
}

func (v Vec3) r() float32 {
  return v.e[0]
}

func (v Vec3) g() float32 {
  return v.e[1]
}

func (v Vec3) b() float32 {
  return v.e[2]
}

func (v Vec3) add(other Vec3) Vec3 {
  return Vec3 {
    e: [3]float32 {
      v.e[0] + other.e[0],
      v.e[1] + other.e[1],
      v.e[2] + other.e[2],
    },
  }
}


func (v Vec3) prod(other Vec3) Vec3 {
  return Vec3 {
    e: [3]float32 {
      v.e[0] * other.e[0],
      v.e[1] * other.e[1],
      v.e[2] * other.e[2],
    },
  }
}

func (v Vec3) sub(other Vec3) Vec3 {
  return Vec3 {
    e: [3]float32 {
      v.e[0] - other.e[0],
      v.e[1] - other.e[1],
      v.e[2] - other.e[2],
    },
  }
}

func (v Vec3) dot(other Vec3) float32 {
  return v.e[0] * other.e[0] +  v.e[1] * other.e[1] +  v.e[2] * other.e[2]
}

func (v Vec3) cross(other Vec3) Vec3 {
  return Vec3 {
    e: [3]float32 {
      v.e[1] * other.e[2] - v.e[2] * other.e[1],
      -v.e[0] * other.e[2] - v.e[2] * other.e[0],
      v.e[0] * other.e[1] - v.e[1] * other.e[0],
    },
  }
}

func (v Vec3) scalar_mult(s float32) Vec3 {
  return vec3(s * v.x(), s * v.y(), s * v.z())
}

func (v Vec3) norm() float32 {
  return float32(math.Sqrt(float64(v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2])))
}

func (v Vec3) normalize() Vec3 {
  l := v.norm()
  return Vec3 {
    e: [3]float32 {
      v.e[0] / l,
      v.e[1] / l,
      v.e[2] / l,
    },
  }
}

func (v Vec3) r2() float32 {
  return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2]
}

func (v Vec3) gamma(g float32) Vec3 {
  return vec3(
    float32(math.Pow(float64(v.e[0]), 1.0 / float64(g))),
    float32(math.Pow(float64(v.e[1]), 1.0 / float64(g))),
    float32(math.Pow(float64(v.e[2]), 1.0 / float64(g))),
  )
}

func color(r Ray, world Hitable, depth int) Vec3 {
  rec := new_hitrec()

  if world.hit(r, 0.001, math.MaxFloat32, &rec) {
    scattered := ray(vec3(0, 0, 0), vec3(0, 0, 0))
    attenuation := vec3(0, 0, 0)

    if (depth < 50 && rec.material.scatter(r, &rec, &attenuation, &scattered)) {
      return attenuation.prod(color(scattered, world, depth + 1))
    } else {
      return vec3(0, 0, 0)
    }
  } else {
    unit_dir := r.direction().normalize()
    t := 0.5 * (unit_dir.y() + 1.0)
    return vec3(1.0, 1.0, 1.0).scalar_mult(1.0 - t).add(vec3(0.5, 0.7, 1.0).scalar_mult(t))
  }
}

func vec3(x, y, z float32) Vec3 {
  return Vec3 {
    e: [3]float32 {x, y, z},
  }
}

/**
  3D Rays
**/

func ray(a, b Vec3) Ray {
  return Ray {
    A: a,
    B: b,
  }
}

func (r Ray) origin() Vec3 { return r.A }

func (r Ray) direction() Vec3 { return r.B }

func (r Ray) point(t float32) Vec3 { return r.origin().add(r.direction().scalar_mult(t)) }

type Ray struct { A, B Vec3 }

func hit_sphere(center Vec3, radius float32, r Ray) float32 {
  oc := r.origin().sub(center)

  a := r.direction().r2()
  b := 2.0 * oc.dot(r.direction())
  c := oc.r2() - radius * radius
  discriminant := b * b - 4 * a * c

  if (discriminant < 0.0) {
    return -1.0
  } else {
    return (-b - float32(math.Sqrt(float64(discriminant)))) / (2.0 * a)
  }
}

type HitRecord struct {
  t float32
  p, normal Vec3
  material Material
}

func new_hitrec() HitRecord {
  return HitRecord {
    t: -1.0,
    p: vec3(0,0,0),
    normal: vec3(0,0,0),
    material: Material {
      mat: NullMaterial,
      albedo: vec3(0,0,0),
    },
  }
}

/* Hitables */

type Hitable interface {
  hit(r Ray, t_min, t_max float32, rec *HitRecord) bool
}

type HitableList []Hitable

func (h HitableList) hit(r Ray, t_min, t_max float32, rec *HitRecord) bool {
  tmp_rec := new_hitrec()
  hit_anything := false
  closest_so_far := t_max

  for i := 0; i < len(h); i++ {
    if h[i].hit(r, t_min, closest_so_far, &tmp_rec) {
      hit_anything = true
      closest_so_far = tmp_rec.t
      *rec = tmp_rec
    }
  }
  return hit_anything
}

/** 3D Spheres **/

func sphere(center Vec3, radius float32, mat Material) Sphere {
  return Sphere {
    center,
    radius,
    mat,
  }
}

func (s Sphere) hit(r Ray, t_min, t_max float32, rec *HitRecord) bool {
  oc := r.origin().sub(s.center)
  a := r.direction().r2()
  b := oc.dot(r.direction())
  c := oc.r2() - s.radius * s.radius
  discriminant := b * b - a * c

  if (discriminant > 0) {
    tmp := (-b - float32(math.Sqrt(float64(discriminant)))) / a;
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.point(rec.t)
      rec.normal = rec.p.sub(s.center).scalar_mult(1.0 / s.radius)
      rec.material = s.material
      return true
    }
    tmp = (-b + float32(math.Sqrt(float64(discriminant)))) / a;
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.point(rec.t)
      rec.normal = rec.p.sub(s.center).scalar_mult(1.0 / s.radius)
      rec.material = s.material
      return true
    }
  }
  return false
}

func random_sphere_point() Vec3 {
  p := vec3(1, 1, 1)

  for p.r2() >= 1.0 {
    p = vec3(rand.Float32(), rand.Float32(), rand.Float32()).scalar_mult(2.0).sub(vec3(1, 1, 1))
  }

  return p
}

type Sphere struct {
  center Vec3
  radius float32
  material Material
}

/* Camera */

type Camera struct {
  origin,
  lower_left_corner,
  horizontal,
  vertical Vec3
}

func (c Camera) get_ray(u, v float32) Ray {
  return Ray {
    A: c.origin,
    B: c.lower_left_corner.add(c.horizontal.scalar_mult(u)).add(c.vertical.scalar_mult(v)).sub(c.origin),
  }
}

/* Materials */

type MaterialType int

const (
  NullMaterial MaterialType = 0
  Lambertian MaterialType = 1
  Metal MaterialType = 2
  Dielectric MaterialType = 3
)

type Material struct {
  mat MaterialType
  albedo Vec3
  fuzz, ref_idx float32
}


func lambertian(ax, ay, az float32) Material {
  return Material {
    mat: Lambertian,
    albedo: vec3(ax, ay, az),
  }
}

func metal(ax, ay, az, fuzz float32) Material {
  return Material {
    mat: Metal,
    albedo: vec3(ax, ay, az),
    fuzz: fuzz,
  }
}

func dielectric(ref_idx float32) Material {
  return Material {
    mat: Dielectric,
    albedo: vec3(1.0, 1.0, 1.0),
    fuzz: 0,
    ref_idx: ref_idx,
  }

}

func (m Material) scatter(r Ray, rec *HitRecord, attenuation *Vec3, scattered *Ray) bool {
  switch(m.mat) {
  case Lambertian:
    target := rec.p.add(rec.normal).add(random_sphere_point())
    *scattered = ray(rec.p, target.sub(rec.p))
    *attenuation = m.albedo
    return true

  case Metal:
    reflected := r.direction().normalize().reflect(rec.normal)
    *scattered = ray(rec.p, reflected.add(random_sphere_point().scalar_mult(m.fuzz)))
    *attenuation = m.albedo
    return scattered.direction().dot(rec.normal) > 0

  case Dielectric:
    outward_normal := vec3(0, 0, 0)
    reflected := r.direction().reflect(rec.normal)
    ni_over_nt := float32(0.0)
    *attenuation = vec3(1, 1, 1)
    refracted := vec3(0, 0, 0)
    var reflect_prob float32
    var cosine float32

    if r.direction().dot(rec.normal) > 0 {
      outward_normal = rec.normal.scalar_mult(-1.0)
      ni_over_nt = m.ref_idx
      cosine = m.ref_idx * r.direction().dot(rec.normal) / r.direction().norm()
    } else {
      outward_normal = rec.normal
      ni_over_nt = float32(1.0 / m.ref_idx)
      cosine = -1.0 * r.direction().dot(rec.normal) / r.direction().norm()
    }

    if r.direction().refract(outward_normal, ni_over_nt, &refracted) {
      reflect_prob = schlick(cosine, m.ref_idx)
    } else {
      *scattered = ray(rec.p, reflected)
      reflect_prob = 1.0
    }

    if (rand.Float32() < reflect_prob) {
      *scattered = ray(rec.p, reflected)
    } else {
      *scattered = ray(rec.p, refracted)
    }
    return true

  default:
    return true
  }
}


func (v Vec3) reflect(n Vec3) Vec3 {
  if (n.norm() - 1.0 > 0.001) {
    panic("reflect: norm must be normalized")
  }
  return v.sub(n.scalar_mult(2 * v.dot(n)))
}

func (v Vec3) refract(n Vec3, ni_over_nt float32, refracted *Vec3) bool {
  uv := v.normalize()
  dt := uv.dot(n)
  discriminant := float64(1.0 - (ni_over_nt * ni_over_nt) * (1 - dt * dt))

  if discriminant > 0 {
    *refracted = uv.sub(n.scalar_mult(dt)).scalar_mult(ni_over_nt).sub(n.scalar_mult(float32(math.Sqrt(discriminant))))
    return true
  } else {
    return false
  }
}

func schlick(cosine float32, ref_idx float32) float32 {
  r0 := (1.0 - ref_idx) / (1.0 + ref_idx)
  r0 = r0 * r0
  return r0 + (1.0 - r0) * float32(math.Pow(float64(1.0 - cosine), 5.0))
}


func main() {
  nx := 400
  ny := 200
  ns := 50

  fmt.Print("P3\n", nx, " ", ny, "\n255\n")

  lower_left_corner := vec3(-2.0, -1.0, -1.0)
  horizontal := vec3(4.0, 0.0, 0.0);
  vertical := vec3(0.0, 2.0, 0.0);
  origin := vec3(0.0, 0.0, 0.0)

  cam := Camera {
    origin: origin,
    lower_left_corner: lower_left_corner,
    horizontal: horizontal,
    vertical: vertical,
  }

  var world HitableList

  world = append(world, sphere(vec3(0, 0, -1), 0.5, lambertian(0.1, 0.2, 0.5)))
  world = append(world, sphere(vec3(0, -100.5, -1), 100, lambertian(0.8, 0.8, 0.0)))
  world = append(world, sphere(vec3(1, 0, -1), 0.5, metal(0.8, 0.6, 0.2, 0.0)))
  world = append(world, sphere(vec3(-1, 0, -1), 0.5, dielectric(1.5)))
  world = append(world, sphere(vec3(-1, 0, -1), -0.45, dielectric(1.5)))

  for j := ny - 1; j >= 0; j-- {
    for i := 0; i < nx; i++ {
      col := vec3(0, 0, 0)

      for s := 0; s < ns; s++ {
        u := (float32(i) + rand.Float32()) / float32(nx);
        v := (float32(j) + rand.Float32()) / float32(ny);
        r := cam.get_ray(u, v)
        col = col.add(color(r, world, 0))
      }

      col = col.scalar_mult(1.0 / float32(ns)).gamma(2.0)

      ir := int(255.99 * col.x());
      ig := int(255.99 * col.y());
      ib := int(255.99 * col.z());

      fmt.Print(ir, " ", ig, " ", ib, "\n")
    }
  }

}
