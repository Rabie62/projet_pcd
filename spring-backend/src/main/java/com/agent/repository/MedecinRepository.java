package com.agent.repository;

import com.agent.model.Medecin;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface MedecinRepository extends JpaRepository<Medecin, Integer> {
    Optional<Medecin> findByUsername(String username);
    Optional<Medecin> findByEmail(String email);
    boolean existsByUsername(String username);
    boolean existsByEmail(String email);
}
